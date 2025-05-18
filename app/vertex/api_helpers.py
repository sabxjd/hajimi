import json
import time
import math
import asyncio
from typing import List, Dict, Any, Callable, Union
from fastapi.responses import JSONResponse, StreamingResponse

from google.auth.transport.requests import Request as AuthRequest
from google.genai import types 
from google import genai # Needed if _execute_gemini_call uses genai.Client directly

# Local module imports
from app.vertex.models import OpenAIRequest, OpenAIMessage # Changed from relative
from app.vertex.message_processing import deobfuscate_text, convert_to_openai_format, convert_chunk_to_openai, create_final_chunk # Changed from relative
import app.vertex.config as app_config # Changed from relative

def create_openai_error_response(status_code: int, message: str, error_type: str) -> Dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": status_code,
            "param": None,
        }
    }

def create_generation_config(request: OpenAIRequest) -> Dict[str, Any]:
    config = {}
    if request.temperature is not None: config["temperature"] = request.temperature
    if request.max_tokens is not None: config["max_output_tokens"] = request.max_tokens
    if request.top_p is not None: config["top_p"] = request.top_p
    if request.top_k is not None: config["top_k"] = request.top_k
    if request.stop is not None: config["stop_sequences"] = request.stop
    if request.seed is not None: config["seed"] = request.seed
    if request.presence_penalty is not None: config["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None: config["frequency_penalty"] = request.frequency_penalty
    if request.n is not None: config["candidate_count"] = request.n
    config["safety_settings"] = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_CIVIC_INTEGRITY", threshold="OFF")
    ]
    return config

def is_response_valid(response):
    if response is None:
        print("DEBUG: Response is None, therefore invalid.")
        return False
    
    # Check for direct text attribute
    if hasattr(response, 'text') and isinstance(response.text, str) and response.text.strip():
        # print("DEBUG: Response valid due to response.text")
        return True
        
    # Check candidates for text content
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates: # Iterate through all candidates
            if hasattr(candidate, 'text') and isinstance(candidate.text, str) and candidate.text.strip():
                # print(f"DEBUG: Response valid due to candidate.text in candidate")
                return True
            if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and isinstance(part.text, str) and part.text.strip():
                        # print(f"DEBUG: Response valid due to part.text in candidate's content part")
                        return True
                        
    # Removed prompt_feedback as a sole criterion for validity.
    # It should only be valid if actual text content is found.
    # Block reasons will be checked explicitly by callers if they need to treat it as an error for retries.
    print("DEBUG: Response is invalid, no usable text content found by is_response_valid.")
    return False

async def fake_stream_generator(client_instance, model_name: str, prompt: Union[types.Content, List[types.Content]], current_gen_config: Dict[str, Any], request_obj: OpenAIRequest, is_auto_attempt: bool):
    response_id = f"chatcmpl-{int(time.time())}"
    async def fake_stream_inner():
        print(f"FAKE STREAMING: Making non-streaming request to Gemini API (Model: {model_name})")
        api_call_task = asyncio.create_task(
            client_instance.aio.models.generate_content(
                model=model_name, contents=prompt, config=current_gen_config
            )
        )
        while not api_call_task.done():
            keep_alive_data = {
                "id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()),
                "model": request_obj.model, "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]
            }
            yield f"data: {json.dumps(keep_alive_data)}\n\n"
            await asyncio.sleep(app_config.FAKE_STREAMING_INTERVAL_SECONDS)
        try:
            response = api_call_task.result()

            # Check for safety blocks first, as this should trigger a retry in auto-mode
            if hasattr(response, 'prompt_feedback') and \
               hasattr(response.prompt_feedback, 'block_reason') and \
               response.prompt_feedback.block_reason:
                block_message = f"Response blocked by safety filter: {response.prompt_feedback.block_reason}"
                if hasattr(response.prompt_feedback, 'block_reason_message') and response.prompt_feedback.block_reason_message:
                    block_message = f"Response blocked by safety filter: {response.prompt_feedback.block_reason_message} (Reason: {response.prompt_feedback.block_reason})"
                print(f"DEBUG: {block_message} (in fake_stream_generator)") # Log this specific condition
                raise ValueError(block_message) # This will be caught by the except Exception as e below it

            if not is_response_valid(response): # is_response_valid now only checks for actual text
                raise ValueError(f"Invalid/empty response in fake stream (no text content): {str(response)[:200]}")
            
            full_text = ""
            if hasattr(response, 'text'):
                full_text = response.text or "" # Coalesce None to empty string
            elif hasattr(response, 'candidates') and response.candidates:
                # Typically, we focus on the first candidate for non-streaming synthesis
                candidate = response.candidates[0]
                if hasattr(candidate, 'text'):
                    full_text = candidate.text or "" # Coalesce None to empty string
                elif hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts:
                    # Ensure parts are iterated and text is joined correctly even if some parts have no text or part.text is None
                    texts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text is not None: # Check part.text exists and is not None
                            texts.append(part.text)
                    full_text = "".join(texts)
            if request_obj.model.endswith("-encrypt-full"):
                full_text = deobfuscate_text(full_text)
            
            chunk_size = max(20, math.ceil(len(full_text) / 10))
            for i in range(0, len(full_text), chunk_size):
                chunk_text = full_text[i:i+chunk_size]
                delta_data = {
                    "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": request_obj.model, "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(delta_data)}\n\n"
                await asyncio.sleep(0.05)
            yield create_final_chunk(request_obj.model, response_id)
            yield "data: [DONE]\n\n"
        except Exception as e:
            err_msg = f"Error in fake_stream_generator: {str(e)}"
            print(err_msg)
            err_resp = create_openai_error_response(500, err_msg, "server_error")
            # It's good practice to log the JSON payload here too for consistency,
            # though the main concern was the true streaming path.
            json_payload_for_fake_stream_error = json.dumps(err_resp)
            # Log the error JSON that WOULD have been sent if not in auto-mode or if this was the final error handler.
            print(f"DEBUG: Internal error in fake_stream_generator. JSON error for handler: {json_payload_for_fake_stream_error}")
            if not is_auto_attempt:
                yield f"data: {json_payload_for_fake_stream_error}\n\n"
                yield "data: [DONE]\n\n"
            raise e # Re-raise the original exception e
    return fake_stream_inner()

async def execute_gemini_call(
    current_client: Any, # Should be genai.Client or similar AsyncClient
    model_to_call: str, 
    prompt_func: Callable[[List[OpenAIMessage]], Union[types.Content, List[types.Content]]], 
    gen_config_for_call: Dict[str, Any],
    request_obj: OpenAIRequest, # Pass the whole request object
    is_auto_attempt: bool = False
):
    actual_prompt_for_call = prompt_func(request_obj.messages)
    
    if request_obj.stream:
        if app_config.FAKE_STREAMING_ENABLED:
            return StreamingResponse(
                await fake_stream_generator(current_client, model_to_call, actual_prompt_for_call, gen_config_for_call, request_obj, is_auto_attempt=is_auto_attempt),
                media_type="text/event-stream"
            )

        response_id_for_stream = f"chatcmpl-{int(time.time())}"
        cand_count_stream = request_obj.n or 1
        
        async def _stream_generator_inner_for_execute(): # Renamed to avoid potential clashes
            try:
                for c_idx_call in range(cand_count_stream):
                    async for chunk_item_call in await current_client.aio.models.generate_content_stream(
                        model=model_to_call, contents=actual_prompt_for_call, config=gen_config_for_call
                    ):
                        yield convert_chunk_to_openai(chunk_item_call, request_obj.model, response_id_for_stream, c_idx_call)
                yield create_final_chunk(request_obj.model, response_id_for_stream, cand_count_stream)
                yield "data: [DONE]\n\n"
            except Exception as e_stream_call:
                print(f"Streaming Error in _execute_gemini_call: {e_stream_call}")
                
                error_message_str = str(e_stream_call)
                # Truncate very long error messages to prevent excessively large JSON payloads.
                if len(error_message_str) > 1024: # Max length for the error string
                    error_message_str = error_message_str[:1024] + "..."
                
                err_resp_content_call = create_openai_error_response(500, error_message_str, "server_error")
                json_payload_for_error = json.dumps(err_resp_content_call)
                # Log the error JSON that WOULD have been sent if not in auto-mode or if this was the final error handler.
                print(f"DEBUG: Internal error in _stream_generator_inner_for_execute. JSON error for handler: {json_payload_for_error}")
                if not is_auto_attempt: # is_auto_attempt is from execute_gemini_call's scope
                    yield f"data: {json_payload_for_error}\n\n"
                    yield "data: [DONE]\n\n"
                raise e_stream_call # Re-raise the original exception
        return StreamingResponse(_stream_generator_inner_for_execute(), media_type="text/event-stream")
    else: 
        response_obj_call = await current_client.aio.models.generate_content(
            model=model_to_call, contents=actual_prompt_for_call, config=gen_config_for_call
        )

        # Check for safety blocks first for non-streaming calls
        if hasattr(response_obj_call, 'prompt_feedback') and \
           hasattr(response_obj_call.prompt_feedback, 'block_reason') and \
           response_obj_call.prompt_feedback.block_reason:
            block_message = f"Response blocked by safety filter: {response_obj_call.prompt_feedback.block_reason}"
            if hasattr(response_obj_call.prompt_feedback, 'block_reason_message') and response_obj_call.prompt_feedback.block_reason_message:
                block_message = f"Response blocked by safety filter: {response_obj_call.prompt_feedback.block_reason_message} (Reason: {response_obj_call.prompt_feedback.block_reason})"
            print(f"DEBUG: {block_message} (in execute_gemini_call non-streaming)") # Log this specific condition
            raise ValueError(block_message)

        if not is_response_valid(response_obj_call): # is_response_valid now only checks for actual text
            raise ValueError("Invalid/empty response from non-streaming Gemini call (no text content).")
        return JSONResponse(content=convert_to_openai_format(response_obj_call, request_obj.model))