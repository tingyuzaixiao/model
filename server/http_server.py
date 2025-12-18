import time
from contextlib import asynccontextmanager

import psutil
from fastapi import FastAPI, APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config.service_config import config
from config.logging_config import logger, access_logger
from server.model.bge_reranker import reranker_model, BGEReranker
from server.schema.request import RerankRequest
from server.schema.response import RerankResponse, HealthResponse, ErrorResponse


# 应用启动时间
startup_time = time.time()

router = APIRouter(
    prefix="/api",
    tags=["model server"]     # 标签，用于API文档分组
)

# 依赖注入：获取模型实例
def get_reranker() -> BGEReranker:
    return reranker_model


# API端点
# include_in_schema: 控制该路由是否包含在自动生成的OpenAPI文档（如Swagger UI）中
@router.get("/", include_in_schema=False)
async def root():
    """根端点"""
    return {
        "message": "BGE-Reranker-v2-M3 服务运行中",
        "version": "1.0.0",
        "docs": "/docs"
    }

@router.get("/health", response_model=HealthResponse)
async def health_check(reranker: BGEReranker = Depends(get_reranker)):
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
            status="healthy" if reranker.is_loaded else "unhealthy",
            model_loaded=reranker.is_loaded,
            model_info=reranker.get_model_info(),
            system_info=system_info,
            service_uptime=system_info["service_uptime"]
        )

    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail="健康检查失败")


@router.post("/rerank", response_model=RerankResponse)
async def rerank_documents(
        request: RerankRequest,
        reranker: BGEReranker = Depends(get_reranker)
):
    """重排序API端点"""
    try:
        if not reranker.is_loaded:
            raise HTTPException(status_code=503, detail="模型未加载，服务不可用")

        # 参数验证
        if request.top_k > len(request.documents):
            request.top_k = len(request.documents)

        # 执行重排序
        ranked_documents, scores, metrics = reranker.rerank(
            query=request.query,
            documents=request.documents,
            top_k=request.top_k,
            batch_size=request.batch_size
        )

        return RerankResponse(
            success=True,
            ranked_documents=ranked_documents,
            scores=scores,
            processing_time=metrics["processing_time"],
            metrics=metrics
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"重排序处理异常: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")


@router.get("/model/info")
async def get_model_info(reranker: BGEReranker = Depends(get_reranker)):
    """获取模型信息"""
    return reranker.get_model_info()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动逻辑
    logger.info("=== 启动重排序服务 ===")
    logger.info(f"服务地址: http://{config.host}:{config.port}")
    logger.info(f"模型名称: {config.model_name}")

    # 加载模型
    success = reranker_model.load_model()
    if not success:
        logger.error("模型加载失败，服务无法启动")
        raise RuntimeError("模型加载失败")

    logger.info("=== 服务启动完成 ===")

    yield  # 这里应用会运行

    # 关闭逻辑
    logger.info("=== 关闭重排序服务 ===")

def init_fastapi() -> FastAPI:
    app = FastAPI(
        title="BGE-Reranker-v2-M3 服务",
        description="军队体检RAG项目重排序API",
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

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """记录访问日志"""
        start_time = time.time()

        try:
            response = await call_next(request)

            process_time = (time.time() - start_time) * 1000
            access_logger.info(
                f"{request.client.host} - \"{request.method} {request.url.path}\" "
                f"{response.status_code} - {process_time:.2f}ms"
            )
            return response
        except Exception as e:
            process_time = (time.time() - start_time) * 1000
            logger.error(f"请求处理异常: {str(e)}", exc_info=True)
            access_logger.error(
                f"{request.client.host} - \"{request.method} {request.url.path}\" "
                f"ERROR - {process_time:.2f}ms - {str(e)}"
            )
            raise

    # 全局异常处理
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """全局异常处理"""
        logger.error(f"未处理的异常: {str(exc)}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                success=False,
                error_message="内部服务器错误",
                error_code="INTERNAL_ERROR"
            ).model_dump()
        )
    return app