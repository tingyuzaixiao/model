import uvicorn

from config.service_config import config

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # ✅ 字符串形式
        host=config.host,
        port=config.port,
        loop="asyncio",
        workers=config.workers,
        log_config=None
    )