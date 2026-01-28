import logging
import logging.handlers
import os
from .service_config import config


def setup_logging():
    """设置日志系统"""

    # 创建日志目录
    log_dir = os.path.dirname(config.log_file)
    os.makedirs(log_dir, exist_ok=True)

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s | %(process)d | %(thread)d | %(levelname)s | [%(filename)s:%(lineno)d] | %(message)s'
    )

    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level.upper()))

    # 清除已有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 主日志文件 - 使用RotatingFileHandler自动滚动
    file_handler = logging.handlers.RotatingFileHandler(
        config.log_file,
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=10,  # 保留10个备份文件
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 添加处理器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 设置第三方库日志级别
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # 设置访问日志
    setup_access_logging()

    return root_logger


def setup_access_logging():
    """设置访问日志"""
    access_log_file = config.log_file.replace('.log', '_access.log')

    access_handler = logging.handlers.RotatingFileHandler(
        access_log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5,
        encoding='utf-8'
    )
    access_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(process)d | %(thread)d | %(levelname)s | [%(filename)s:%(lineno)d] | %(message)s'
    ))

    access_logger = logging.getLogger('access')
    access_logger.setLevel(logging.INFO)
    access_logger.propagate = False

    # 清除已有处理器
    for handler in access_logger.handlers[:]:
        access_logger.removeHandler(handler)

    access_logger.addHandler(access_handler)


# 初始化日志
logger = setup_logging()
access_logger = logging.getLogger('access')