from typing import Optional
from fastapi import HTTPException, Header, Query, Depends
import app.config.settings as settings
import base64
import json

# 自定义密码校验依赖函数
async def custom_verify_password(
    authorization: Optional[str] = Header(None, description="OpenAI 格式请求 Key, 格式: Bearer sk-xxxx"),
    x_goog_api_key: Optional[str] = Header(None, description="Gemini 格式请求 Key, 从请求头 x-goog-api-key 获取"),
    key: Optional[str] = Query(None, description="Gemini 格式请求 Key, 从查询参数 key 获取"),
    x_ip_token: Optional[str] = Header(None, alias="x-ip-token", description="HuggingFace 格式请求 token"),
    alt: Optional[str] = None
):
    """
    校验 API Key。
    1. 从请求中提取客户端提供的 API Key 。
    2. 根据类型，与项目配置的密钥进行比对。
    3. 如果 Key 无效、缺失或不匹配，则抛出 HTTPException。
    """
    # 如果启用HuggingFace模式，使用x-ip-token进行验证
    if settings.HUGGINGFACE and x_ip_token:
        try:
            # JWT 通常是 header.payload.signature 的格式
            # 我们只关心 payload 部分（也就是用'.'分隔后的中间那部分）
            parts = x_ip_token.split('.')
            if len(parts) < 2:  # 一个有效的JWT至少要有头部和载荷部分
                raise ValueError("无效的JWT格式：组成部分不足以提取payload。")

            payload_encoded = parts[1]  # 获取Base64Url编码的payload字符串

            # 对payload进行Base64Url解码
            # Base64Url编码可能会省略末尾的'='填充，解码时需要补上
            payload_encoded += '=' * (-len(payload_encoded) % 4)
            
            # 使用urlsafe_b64decode进行解码
            decoded_payload_bytes = base64.urlsafe_b64decode(payload_encoded)
            # 将解码后的字节串转换为UTF-8字符串，然后解析为JSON对象
            payload = json.loads(decoded_payload_bytes.decode('utf-8'))

            # 检查解码后 payload 中的 'error' 字段
            error_in_token = payload.get("error")  # 使用 .get() 可以安全地获取字段值，如果字段不存在则返回 None

            if error_in_token == "InvalidAccessToken":
                # 如果错误是 "InvalidAccessToken"，则抛出403禁止访问异常
                raise HTTPException(
                    status_code=403,  # 禁止访问 - token有效，但表明凭证无效或权限不足
                    detail="访问被拒绝：x-ip-token 表明 'InvalidAccessToken'。"
                )
            elif error_in_token is None:  # JSON 中的 'null' 在Python中会被解析为 None
                # 如果 error 是 null (None)，则视为认证成功
                return settings.HUGGINGFACE_API_KEY
            else:
                # 如果 'error' 字段是其他非 None 也非 "InvalidAccessToken" 的字符串
                raise HTTPException(
                    status_code=403,  # 禁止访问
                    detail=f"访问被拒绝：x-ip-token 表明存在无法识别的错误：'{error_in_token}'。"
                )
        except ValueError as ve:  # 主要捕获JWT格式相关的错误
            raise HTTPException(
                status_code=400,  # 错误请求 - 客户端发送的token格式有问题
                detail=f"x-ip-token 中的JWT格式无效: {str(ve)}"
            )
        except (json.JSONDecodeError, base64.binascii.Error, UnicodeDecodeError) as e:
            # 捕获Base64解码错误或JSON解析错误
            raise HTTPException(
                status_code=400,  # 错误请求 - 客户端发送的token内容有问题
                detail=f"x-ip-token payload 格式错误，无法解码或解析: {str(e)}"
            )
        except Exception as e:
            # 捕获在处理token过程中其他未预料到的异常
            raise HTTPException(
                status_code=500,  # 服务器内部错误
                detail="处理 x-ip-token 时发生内部错误。"
            )
    
    # 原有的验证逻辑
    client_provided_api_key: Optional[str] = None

    # 提取客户端提供的 Key 
    if x_goog_api_key: 
        client_provided_api_key = x_goog_api_key
    elif key:
        client_provided_api_key = key
    elif authorization and authorization.startswith("Bearer "): 
        token = authorization.split(" ", 1)[1]
        client_provided_api_key = token

    # 进行校验和比对
    if (not client_provided_api_key) or (client_provided_api_key != settings.PASSWORD) :
            raise HTTPException(
                status_code=401, detail="Unauthorized: Invalid token")

def verify_web_password(password:str):
    if password != settings.WEB_PASSWORD:
        return False
    return True