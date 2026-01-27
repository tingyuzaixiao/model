import time
from typing import Dict, Any

from FlagEmbedding import BGEM3FlagModel

from config.logging_config import logger
from config.service_config import config


class BGEM3:
    QUERY_MAX_LENGTH = 512
    """BGE重排序模型封装"""

    def __init__(self):
        self.logger = logger.getChild('bge-m3')
        self.model = None
        self.model_path = config.model_path
        self.devices = config.devices
        self.is_loaded = False
        self.load_time = 0
        self.total_queries = 0
        self.total_processing_time = 0

    def load_model(self) -> bool:
        """加载模型"""
        try:
            self.logger.info(f"开始加载模型: {self.model_path}")
            start_time = time.time()

            self.model = BGEM3FlagModel(self.model_path,
                                   use_fp16=True,
                                   devices=self.devices,
                                   query_max_length=BGEM3.QUERY_MAX_LENGTH,
                                   return_dense=True,
                                   return_sparse=True)

            self.load_time = time.time() - start_time
            self.is_loaded = True

            self.logger.info(f"模型加载成功: {self.model_path}")
            self.logger.info(f"加载耗时: {self.load_time:.2f}秒")
            self.logger.info(f"模型设备: {self.devices}, 最大长度: {BGEM3.QUERY_MAX_LENGTH}")

            # 预热模型
            self._warmup()

            return True

        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}", exc_info=True)
            self.is_loaded = False
            return False

    def _warmup(self):
        """预热模型"""
        self.logger.info("开始模型预热...")
        sentences = ["如何使用vLLM进行模型部署？"]

        try:
            embeddings = self.model.encode(sentences, return_dense=True, return_sparse=True)
            self.logger.info("模型预热完成")
        except Exception as e:
            self.logger.warning(f"模型预热失败: {str(e)}")

    def embeddings(self, query: str) -> dict:
        """执行重排序"""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("模型未加载")

        if not query:
            raise RuntimeError("query is empty")

        try:
            return self.model.encode([query], return_dense=True, return_sparse=True)
        except Exception as e:
            self.logger.error(f"重排序处理失败: {str(e)}", exc_info=True)
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "devices": self.devices,
        }


# 全局模型实例
embeddings_model = BGEM3()