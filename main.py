from config.service_config import config
from server.http_server import init_fastapi

if __name__ == "__main__":
    app = init_fastapi()

    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_config=None  # 使用自定义日志配置
    )