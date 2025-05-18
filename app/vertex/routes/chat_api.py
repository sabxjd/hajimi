import asyncio
import json # Needed for error streaming
import random
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any

from app.utils.logging import vertex_log

# Google and OpenAI specific imports
from google.genai import types
from google import genai
import openai
from app.vertex.credentials_manager import _refresh_auth, CredentialManager

# Local module imports
from app.vertex.models import OpenAIRequest, OpenAIMessage
from app.vertex.auth import get_api_key
import app.vertex.config as app_config
from app.vertex.model_loader import get_vertex_models, get_vertex_express_models
from app.vertex.message_processing import (
    create_gemini_prompt,
    create_encrypted_gemini_prompt,
    create_encrypted_full_gemini_prompt
)
from app.vertex.api_helpers import (
    create_generation_config,
    create_openai_error_response,
    execute_gemini_call
)

router = APIRouter()

@router.post("/v1/chat/completions")
async def chat_completions(fastapi_request: Request, request: OpenAIRequest, api_key: str = Depends(get_api_key)):
    try:
        # 获取credential_manager，如果不存在则创建一个新的
        try:
            credential_manager_instance = fastapi_request.app.state.credential_manager
            vertex_log('info', "Using existing credential manager from app state")
        except AttributeError:
            # 如果app.state中没有credential_manager，则创建一个新的
            vertex_log('warning', "No credential_manager found in app.state, creating a new one")
            credential_manager_instance = CredentialManager()
        
        OPENAI_DIRECT_SUFFIX = "-openai"
        EXPERIMENTAL_MARKER = "-exp-"
        PAY_PREFIX = "[PAY]"
        EXPRESS_PREFIX = "[EXPRESS] " # Note the space for easier stripping
        
        # Model validation based on a predefined list has been removed as per user request.
        # The application will now attempt to use any provided model string.
        # We still need to fetch vertex_express_model_ids for the Express Mode logic.
        # vertex_express_model_ids = await get_vertex_express_models() # We'll use the prefix now

        # Updated logic for is_openai_direct_model
        is_openai_direct_model = False
        if request.model.endswith(OPENAI_DIRECT_SUFFIX):
            temp_name_for_marker_check = request.model[:-len(OPENAI_DIRECT_SUFFIX)]
            if temp_name_for_marker_check.startswith(PAY_PREFIX):
                is_openai_direct_model = True
            elif EXPERIMENTAL_MARKER in temp_name_for_marker_check:
                is_openai_direct_model = True
        is_auto_model = request.model.endswith("-auto")
        is_grounded_search = request.model.endswith("-search")
        is_encrypted_model = request.model.endswith("-encrypt")
        is_encrypted_full_model = request.model.endswith("-encrypt-full")
        is_nothinking_model = request.model.endswith("-nothinking")
        is_max_thinking_model = request.model.endswith("-max")
        base_model_name = request.model # Start with the full model name

        # Determine base_model_name by stripping known prefixes and suffixes
        # Order of stripping: Prefixes first, then suffixes.
        
        is_express_model_request = False
        if base_model_name.startswith(EXPRESS_PREFIX):
            is_express_model_request = True
            base_model_name = base_model_name[len(EXPRESS_PREFIX):]

        if base_model_name.startswith(PAY_PREFIX):
            base_model_name = base_model_name[len(PAY_PREFIX):]

        # Suffix stripping (applied to the name after prefix removal)
        # This order matters if a model could have multiple (e.g. -encrypt-auto, though not currently a pattern)
        if is_openai_direct_model: # This check is based on request.model, so it's fine here
            # If it was an OpenAI direct model, its base name is request.model minus suffix.
            # We need to ensure PAY_PREFIX or EXPRESS_PREFIX are also stripped if they were part of the original.
            temp_base_for_openai = request.model[:-len(OPENAI_DIRECT_SUFFIX)]
            if temp_base_for_openai.startswith(EXPRESS_PREFIX):
                temp_base_for_openai = temp_base_for_openai[len(EXPRESS_PREFIX):]
            if temp_base_for_openai.startswith(PAY_PREFIX):
                temp_base_for_openai = temp_base_for_openai[len(PAY_PREFIX):]
            base_model_name = temp_base_for_openai # Assign the fully stripped name
        elif is_auto_model: base_model_name = base_model_name[:-len("-auto")]
        elif is_grounded_search: base_model_name = base_model_name[:-len("-search")]
        elif is_encrypted_full_model: base_model_name = base_model_name[:-len("-encrypt-full")] # Must be before -encrypt
        elif is_encrypted_model: base_model_name = base_model_name[:-len("-encrypt")]
        elif is_nothinking_model: base_model_name = base_model_name[:-len("-nothinking")]
        elif is_max_thinking_model: base_model_name = base_model_name[:-len("-max")]
        
        # Specific model variant checks (if any remain exclusive and not covered dynamically)
        if is_nothinking_model and base_model_name != "gemini-2.5-flash-preview-04-17":
            return JSONResponse(status_code=400, content=create_openai_error_response(400, f"Model '{request.model}' (-nothinking) is only supported for 'gemini-2.5-flash-preview-04-17'.", "invalid_request_error"))
        if is_max_thinking_model and base_model_name != "gemini-2.5-flash-preview-04-17":
            return JSONResponse(status_code=400, content=create_openai_error_response(400, f"Model '{request.model}' (-max) is only supported for 'gemini-2.5-flash-preview-04-17'.", "invalid_request_error"))

        generation_config = create_generation_config(request)

        client_to_use = None
        express_api_keys_list = app_config.VERTEX_EXPRESS_API_KEY_VAL

        # This client initialization logic is for Gemini models.
        # OpenAI Direct models have their own client setup and will return before this.
        if is_openai_direct_model:
            # OpenAI Direct logic is self-contained and will return.
            # If it doesn't return, it means we proceed to Gemini logic, which shouldn't happen
            # if is_openai_direct_model is true. The main if/elif/else for model types handles this.
            pass
        elif is_express_model_request:
            if not express_api_keys_list:
                error_msg = f"Model '{request.model}' is an Express model and requires an Express API key, but none are configured."
                vertex_log('error', error_msg)
                return JSONResponse(status_code=401, content=create_openai_error_response(401, error_msg, "authentication_error"))

            vertex_log('info', f"INFO: Attempting Vertex Express Mode for model request: {request.model} (base: {base_model_name})")
            indexed_keys = list(enumerate(express_api_keys_list))
            random.shuffle(indexed_keys)
            
            for original_idx, key_val in indexed_keys:
                try:
                    client_to_use = genai.Client(vertexai=True, api_key=key_val)
                    vertex_log('info', f"INFO: Using Vertex Express Mode for model {request.model} (base: {base_model_name}) with API key (original index: {original_idx}).")
                    break # Successfully initialized client
                except Exception as e:
                    vertex_log('warning', f"WARNING: Vertex Express Mode client init failed for API key (original index: {original_idx}) for model {request.model}: {e}. Trying next key.")
                    client_to_use = None # Ensure client_to_use is None for this attempt

            if client_to_use is None: # All configured Express keys failed
                error_msg = f"All configured Express API keys failed to initialize for model '{request.model}'."
                vertex_log('error', error_msg)
                return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))
        
        else: # Not an Express model request, therefore an SA credential model request for Gemini
            vertex_log('info', f"INFO: Model '{request.model}' is an SA credential request for Gemini. Attempting SA credentials.")
            rotated_credentials, rotated_project_id = credential_manager_instance.get_random_credentials()
            
            if rotated_credentials and rotated_project_id:
                try:
                    client_to_use = genai.Client(vertexai=True, credentials=rotated_credentials, project=rotated_project_id, location="global")
                    vertex_log('info', f"INFO: Using SA credential for Gemini model {request.model} (project: {rotated_project_id})")
                except Exception as e:
                    client_to_use = None # Ensure it's None on failure
                    error_msg = f"SA credential client initialization failed for Gemini model '{request.model}': {e}."
                    vertex_log('error', error_msg)
                    return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))
            else: # No SA credentials available for an SA model request
                error_msg = f"Model '{request.model}' requires SA credentials for Gemini, but none are available or loaded."
                vertex_log('error', error_msg)
                return JSONResponse(status_code=401, content=create_openai_error_response(401, error_msg, "authentication_error"))

        # If we reach here and client_to_use is still None, it means it's an OpenAI Direct Model,
        # which handles its own client and responses.
        # For Gemini models (Express or SA), client_to_use must be set, or an error returned above.
        if not is_openai_direct_model and client_to_use is None:
             # This case should ideally not be reached if the logic above is correct,
             # as each path (Express/SA for Gemini) should either set client_to_use or return an error.
             # This is a safeguard.
            vertex_log('critical', f"CRITICAL ERROR: Client for Gemini model '{request.model}' was not initialized, and no specific error was returned. This indicates a logic flaw.")
            return JSONResponse(status_code=500, content=create_openai_error_response(500, "Critical internal server error: Gemini client not initialized.", "server_error"))

        encryption_instructions_placeholder = ["// Protocol Instructions Placeholder //"] # Actual instructions are in message_processing
        if is_openai_direct_model:
            vertex_log('info', f"INFO: Using OpenAI Direct Path for model: {request.model}")
            # This mode exclusively uses rotated credentials, not express keys.
            rotated_credentials, rotated_project_id = credential_manager_instance.get_random_credentials()

            if not rotated_credentials or not rotated_project_id:
                error_msg = "OpenAI Direct Mode requires GCP credentials, but none were available or loaded successfully."
                vertex_log('error', error_msg)
                return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))

            vertex_log('info', f"INFO: [OpenAI Direct Path] Using credentials for project: {rotated_project_id}")
            gcp_token = _refresh_auth(rotated_credentials)

            if not gcp_token:
                error_msg = f"Failed to obtain valid GCP token for OpenAI client (Source: Credential Manager, Project: {rotated_project_id})."
                vertex_log('error', error_msg)
                return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))

            PROJECT_ID = rotated_project_id
            LOCATION = "global" # Fixed as per user confirmation
            VERTEX_AI_OPENAI_ENDPOINT_URL = (
                f"https://aiplatform.googleapis.com/v1beta1/"
                f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi"
            )
            # base_model_name is already extracted (e.g., "gemini-1.5-pro-exp-v1")
            UNDERLYING_MODEL_ID = f"google/{base_model_name}"

            openai_client = openai.AsyncOpenAI(
                base_url=VERTEX_AI_OPENAI_ENDPOINT_URL,
                api_key=gcp_token, # OAuth token
            )

            openai_safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
                {"category": 'HARM_CATEGORY_CIVIC_INTEGRITY', "threshold": 'OFF'}
            ]

            openai_params = {
                "model": UNDERLYING_MODEL_ID,
                "messages": [msg.model_dump(exclude_unset=True) for msg in request.messages],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
                "stream": request.stream,
                "stop": request.stop,
                "seed": request.seed,
                "n": request.n,
            }
            openai_params = {k: v for k, v in openai_params.items() if v is not None}

            openai_extra_body = {
                'google': {
                    'safety_settings': openai_safety_settings
                }
            }

            if request.stream:
                async def openai_stream_generator():
                    try:
                        stream_response = await openai_client.chat.completions.create(
                            **openai_params,
                            extra_body=openai_extra_body
                        )
                        async for chunk in stream_response:
                            try:
                                yield f"data: {chunk.model_dump_json()}\n\n"
                            except Exception as chunk_serialization_error:
                                error_msg_chunk = f"Error serializing OpenAI chunk for {request.model}: {str(chunk_serialization_error)}. Chunk: {str(chunk)[:200]}"
                                vertex_log('error', error_msg_chunk)
                                # Truncate
                                if len(error_msg_chunk) > 1024:
                                    error_msg_chunk = error_msg_chunk[:1024] + "..."
                                error_response_chunk = create_openai_error_response(500, error_msg_chunk, "server_error")
                                json_payload_for_chunk_error = json.dumps(error_response_chunk)
                                vertex_log('debug', f"DEBUG: Yielding chunk serialization error JSON payload (OpenAI path): {json_payload_for_chunk_error}")
                                yield f"data: {json_payload_for_chunk_error}\n\n"
                                yield "data: [DONE]\n\n"
                                return # Stop further processing for this request
                        yield "data: [DONE]\n\n"
                    except Exception as stream_error:
                        original_error_message = str(stream_error)
                        # Truncate very long error messages
                        if len(original_error_message) > 1024:
                            original_error_message = original_error_message[:1024] + "..."
                        
                        error_msg_stream = f"Error during OpenAI client streaming for {request.model}: {original_error_message}"
                        vertex_log('error', error_msg_stream)
                        
                        error_response_content = create_openai_error_response(500, error_msg_stream, "server_error")
                        json_payload_for_stream_error = json.dumps(error_response_content)
                        vertex_log('debug', f"DEBUG: Yielding stream error JSON payload (OpenAI path): {json_payload_for_stream_error}")
                        yield f"data: {json_payload_for_stream_error}\n\n"
                        yield "data: [DONE]\n\n"
                return StreamingResponse(openai_stream_generator(), media_type="text/event-stream")
            else: # Not streaming
                try:
                    response = await openai_client.chat.completions.create(
                        **openai_params,
                        extra_body=openai_extra_body
                    )
                    return JSONResponse(content=response.model_dump(exclude_unset=True))
                except Exception as generate_error:
                    error_msg_generate = f"Error calling OpenAI client for {request.model}: {str(generate_error)}"
                    vertex_log('error', error_msg_generate)
                    error_response = create_openai_error_response(500, error_msg_generate, "server_error")
                    return JSONResponse(status_code=500, content=error_response)
        elif is_auto_model:
            vertex_log('info', f"Processing auto model: {request.model}")
            attempts = [
                {"name": "base", "model": base_model_name, "prompt_func": create_gemini_prompt, "config_modifier": lambda c: c},
                {"name": "encrypt", "model": base_model_name, "prompt_func": create_encrypted_gemini_prompt, "config_modifier": lambda c: {**c, "system_instruction": encryption_instructions_placeholder}},
                {"name": "old_format", "model": base_model_name, "prompt_func": create_encrypted_full_gemini_prompt, "config_modifier": lambda c: c}                  
            ]
            last_err = None
            for attempt in attempts:
                vertex_log('info', f"Auto-mode attempting: '{attempt['name']}' for model {attempt['model']}")
                current_gen_config = attempt["config_modifier"](generation_config.copy())
                try:
                    # Pass is_auto_attempt=True for auto-mode calls
                    return await execute_gemini_call(client_to_use, attempt["model"], attempt["prompt_func"], current_gen_config, request, is_auto_attempt=True)
                except Exception as e_auto:
                    last_err = e_auto
                    vertex_log('info', f"Auto-attempt '{attempt['name']}' for model {attempt['model']} failed: {e_auto}")
                    await asyncio.sleep(1)
            
            vertex_log('info', f"All auto attempts failed. Last error: {last_err}")
            err_msg = f"All auto-mode attempts failed for model {request.model}. Last error: {str(last_err)}"
            if not request.stream and last_err:
                 return JSONResponse(status_code=500, content=create_openai_error_response(500, err_msg, "server_error"))
            elif request.stream: 
                # This is the final error handling for auto-mode if all attempts fail AND it was a streaming request
                async def final_auto_error_stream():
                    err_content = create_openai_error_response(500, err_msg, "server_error")
                    json_payload_final_auto_error = json.dumps(err_content)
                    # Log the final error being sent to client after all auto-retries failed
                    vertex_log('debug', f"DEBUG: Auto-mode all attempts failed. Yielding final error JSON: {json_payload_final_auto_error}")
                    yield f"data: {json_payload_final_auto_error}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(final_auto_error_stream(), media_type="text/event-stream")
            return JSONResponse(status_code=500, content=create_openai_error_response(500, "All auto-mode attempts failed without specific error.", "server_error"))

        else: # Not an auto model
            current_prompt_func = create_gemini_prompt
            # Determine the actual model string to call the API with (e.g., "gemini-1.5-pro-search")
            api_model_string = request.model 

            if is_grounded_search:
                search_tool = types.Tool(google_search=types.GoogleSearch())
                generation_config["tools"] = [search_tool]
            elif is_encrypted_model:
                generation_config["system_instruction"] = encryption_instructions_placeholder
                current_prompt_func = create_encrypted_gemini_prompt
            elif is_encrypted_full_model:
                generation_config["system_instruction"] = encryption_instructions_placeholder
                current_prompt_func = create_encrypted_full_gemini_prompt
            elif is_nothinking_model:
                generation_config["thinking_config"] = {"thinking_budget": 0}
            elif is_max_thinking_model:
                generation_config["thinking_config"] = {"thinking_budget": 24576}
            
            # For non-auto models, the 'base_model_name' might have suffix stripped.
            # We should use the original 'request.model' for API call if it's a suffixed one,
            # or 'base_model_name' if it's truly a base model without suffixes.
            # The current logic uses 'base_model_name' for the API call in the 'else' block.
            # This means if `request.model` was "gemini-1.5-pro-search", `base_model_name` becomes "gemini-1.5-pro"
            # but the API call might need the full "gemini-1.5-pro-search".
            # Let's use `request.model` for the API call here, and `base_model_name` for checks like Express eligibility.
            # For non-auto mode, is_auto_attempt defaults to False in execute_gemini_call
            return await execute_gemini_call(client_to_use, base_model_name, current_prompt_func, generation_config, request)

    except Exception as e:
        error_msg = f"Unexpected error in chat_completions endpoint: {str(e)}"
        vertex_log('error', error_msg)
        return JSONResponse(status_code=500, content=create_openai_error_response(500, error_msg, "server_error"))