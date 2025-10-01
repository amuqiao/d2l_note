import datetime
import functools

def time_it(func):
    """
    统计函数运行时间的装饰器
    
    功能: 记录函数的开始时间、结束时间，计算并打印运行时间
    """
    @functools.wraps(func)  # 保留被装饰函数的元信息
    def wrapper(*args, **kwargs):
        # 记录开始时间
        start_time = datetime.datetime.now()
        
        # 执行被装饰的函数
        result = func(*args, **kwargs)
        
        # 记录结束时间
        end_time = datetime.datetime.now()
        
        # 计算运行时间
        elapsed_time = end_time - start_time
        
        # 打印统计信息
        print(f"函数 {func.__name__} 执行信息:")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        print(f"运行时间: {elapsed_time.total_seconds():.6f} 秒")
        print("-" * 50)
        
        # 返回函数执行结果
        return result
    
    return wrapper

# 示例用法
if __name__ == "__main__":
    @time_it
    def test_function(seconds):
        """测试函数，模拟耗时操作"""
        import time
        time.sleep(seconds)
        return f"完成 {seconds} 秒的等待"
    
    # 测试不同运行时间的函数
    test_function(1)
    test_function(0.5)
    
    @time_it
    def calculate_sum(n):
        """计算1到n的和"""
        return sum(range(n+1))
    
    result = calculate_sum(1000000)
    print(f"计算结果: {result}")
