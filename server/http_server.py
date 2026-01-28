import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import psutil
from fastapi import FastAPI, APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config.service_config import config
from config.logging_config import logger, access_logger
from server.model.bge_m3 import BGEM3, embeddings_model
from server.schema.request import EmbeddingsRequest
from server.schema.response import HealthResponse, ErrorResponse, EmbeddingResponse
from server.tool.atomic_counter import AtomicCounter

import numpy as np

# 应用启动时间
startup_time = time.time()

MAX_CONCURRENT = 16

thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT)

counter = AtomicCounter(0)

router = APIRouter(
    prefix="/api",
    tags=["model server"]     # 标签，用于API文档分组
)

# 依赖注入：获取模型实例
def get_embeddings() -> BGEM3:
    return embeddings_model

def model_inference(query: str, embeddings_obj: BGEM3) -> dict:
    return embeddings_obj.embeddings(query=query)


# API端点
# include_in_schema: 控制该路由是否包含在自动生成的OpenAPI文档（如Swagger UI）中
@router.get("/", include_in_schema=False)
async def root():
    """根端点"""
    return {
        "message": "BGE-M3 服务运行中",
        "version": "1.0.0",
    }

@router.get("/health", response_model=HealthResponse)
async def health_check(embeddings_obj: BGEM3 = Depends(get_embeddings)):
    """健康检查端点"""
    try:
        # 系统信息
        process = psutil.Process()
        memory_info = process.memory_info()

        system_info = {
            "memory_used_mb": round(memory_info.rss / 1024 / 1024, 2),
            "cpu_percent": round(process.cpu_percent(), 2),
            "service_uptime": round(time.time() - startup_time, 2)
        }

        return HealthResponse(
            status="healthy" if embeddings_obj.is_loaded else "unhealthy",
            model_loaded=embeddings_obj.is_loaded,
            model_info=embeddings_obj.get_model_info(),
            system_info=system_info,
            service_uptime=system_info["service_uptime"]
        )

    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail="健康检查失败")


@router.post("/embeddings", response_model=EmbeddingResponse)
async def embeddings(
        request: EmbeddingsRequest,
        embeddings_obj: BGEM3 = Depends(get_embeddings)
):
    current_count = counter.increment()
    try:
        if current_count > MAX_CONCURRENT * 2:
            raise HTTPException(status_code=503, detail=f"请求并发数过高：{current_count}")

        if not embeddings_obj.is_loaded:
            raise HTTPException(status_code=503, detail="模型未加载，服务不可用")

        loop = asyncio.get_event_loop()
        dict_obj = await loop.run_in_executor(thread_pool,
                                            model_inference,
                                            request.query, embeddings_obj)
        dense_vec = dict_obj["dense_vecs"][0].astype(np.float32).tolist()
        lexical_weights = {
            int(k): float(v)
            for k, v in dict_obj["lexical_weights"][0].items()
        }

        return EmbeddingResponse(
            code=0,
            dense_vec=dense_vec,
            lexical_weights=lexical_weights
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"embeddings异常: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")
    finally:
        counter.decrement()


@router.get("/model/info")
async def get_model_info(embeddings_obj: BGEM3 = Depends(get_embeddings)):
    """获取模型信息"""
    return embeddings_obj.get_model_info()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动逻辑
    logger.info("=== 启动embeddings服务 ===")
    logger.info(f"服务地址: http://{config.host}:{config.port}")
    logger.info(f"模型地址: {config.model_path}")

    # 加载模型
    success = embeddings_model.load_model()
    if not success:
        logger.error("模型加载失败，服务无法启动")
        raise RuntimeError("模型加载失败")

    logger.info("=== 服务启动完成 ===")

    yield  # 这里应用会运行

    # 关闭逻辑
    logger.info("=== 关闭embeddings服务 ===")

def init_fastapi() -> FastAPI:
    loop = asyncio.get_event_loop()
    loop.set_default_executor(
        ThreadPoolExecutor(max_workers=MAX_CONCURRENT + 2)  # +2 为日志/中间件留余量
    )

    app = FastAPI(
        title="BGE-M3 服务",
        description="embeddings API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan  # 使用lifespan替代startup/shutdown事件
    )
    app.include_router(router)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # @app.middleware("http")
    # async def log_requests(request: Request, call_next):
    #     """记录访问日志"""
    #     start_time = time.time()
    #
    #     try:
    #         response = await call_next(request)
    #
    #         process_time = (time.time() - start_time) * 1000
    #         access_logger.info(
    #             f"{request.client.host} - \"{request.method} {request.url.path}\" "
    #             f"{response.status_code} - {process_time:.2f}ms"
    #         )
    #         return response
    #     except Exception as e:
    #         process_time = (time.time() - start_time) * 1000
    #         logger.error(f"请求处理异常: {str(e)}", exc_info=True)
    #         access_logger.error(
    #             f"{request.client.host} - \"{request.method} {request.url.path}\" "
    #             f"ERROR - {process_time:.2f}ms - {str(e)}"
    #         )
    #         raise

    # 全局异常处理
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """全局异常处理"""
        logger.error(f"未处理的异常: {str(exc)}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error_message="内部服务器错误",
                error_code="INTERNAL_ERROR"
            ).model_dump()
        )
    return app