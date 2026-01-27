import threading
import time
import random


class AtomicCounter:
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()

    def increment(self, delta = 1) -> int:
        with self._lock:
            self._value += delta
            return self._value

    def decrement(self, delta = 1) -> int:
        with self._lock:
            self._value -= delta
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


if __name__ == "__main__":
    # 使用示例
    counter = AtomicCounter(0)


    def simulated_task(worker_id):
        """模拟一个耗时任务"""
        # 任务开始，计数加1
        current_count = counter.increment()
        print(f"Worker {worker_id} 开始任务。当前并发任务数: {current_count}")

        # 模拟任务处理时间
        time.sleep(random.uniform(0.5, 2.0))

        # 任务结束，计数减1
        current_count = counter.decrement()
        print(f"Worker {worker_id} 结束任务。当前并发任务数: {current_count}")


    # 创建并启动多个工作线程
    threads = []
    for i in range(5):
        t = threading.Thread(target=simulated_task, args=(i,))
        threads.append(t)
        t.start()

    # 等待所有线程执行完毕
    for t in threads:
        t.join()

    print(f"所有任务完成。最终计数器值: {counter.value}")  # 预期为0