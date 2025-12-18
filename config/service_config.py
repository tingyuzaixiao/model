import os
from dataclasses import dataclass


@dataclass
class ServiceConfig:
    """服务配置"""
    # 服务设置
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # 模型设置
    # model_name两种写法都是对的
    # model_name: str = "BAAI/bge-reranker-v2-m3"
    model_name: str = "/home/zhangjiang/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    max_length: int = 512
    batch_size: int = 32
    device: str = "cuda"
    use_fp16: bool = True

    # 日志配置
    log_level: str = "INFO"
    log_file: str = "/opt/reranker-service/logs/reranker_service.log"

    def __post_init__(self):
        """初始化后处理"""
        # 自动检测设备
        try:
            import torch
            if not torch.cuda.is_available():
                self.device = "cpu"
                self.use_fp16 = False
        except ImportError:
            self.device = "cpu"
            self.use_fp16 = False

        # 从环境变量加载配置
        # self._load_from_env()

    def _load_from_env(self):
        """从环境变量加载配置"""
        if os.getenv("RERANKER_HOST"):
            self.host = os.getenv("RERANKER_HOST")
        if os.getenv("RERANKER_PORT"):
            self.port = int(os.getenv("RERANKER_PORT"))
        if os.getenv("MODEL_NAME"):
            self.model_name = os.getenv("MODEL_NAME")
        if os.getenv("LOG_LEVEL"):
            self.log_level = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_FILE"):
            self.log_file = os.getenv("LOG_FILE")
        if os.getenv("CUDA_VISIBLE_DEVICES"):
            self.device = "cuda"

# 全局配置实例
config = ServiceConfig()