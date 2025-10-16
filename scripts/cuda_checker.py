#!/usr/bin/env python3
"""
CUDA可用性检查器

功能: 检查系统中的CUDA支持情况、PyTorch配置以及GPU设备信息，
帮助用户诊断深度学习环境的GPU加速可行性。

参数:
  -v, --verbose   启用详细输出模式，显示更多调试信息
  -h, --help      显示此帮助信息并退出

用法示例:
  基本检查:
    python cuda_checker.py
  
  详细检查:
    python cuda_checker.py -v
"""

import torch
import sys
import datetime
import subprocess
import platform
import argparse


class CudaChecker:
    """
    CUDA可用性检查器，提供优雅的接口来检查和显示CUDA相关信息
    """

    def __init__(self, verbose=False):
        """初始化检查器，记录开始时间"""
        self.verbose = verbose
        self.start_time = datetime.datetime.now()
        self.end_time = None
        self.cuda_available = False
        self.device_count = 0

    def _print_header(self):
        """打印检查头部信息"""
        print("=" * 60)
        print(
            f"🔍 CUDA 可用性检查 - 开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if self.verbose:
            print("📝 详细输出模式已启用")
        print("=" * 60)

    def _print_footer(self):
        """打印检查尾部信息"""
        self.end_time = datetime.datetime.now()
        elapsed_time = (self.end_time - self.start_time).total_seconds()

        print("=" * 60)
        print(
            f"✅ CUDA 可用性检查完成 - 结束时间: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"⏱️  检查耗时: {elapsed_time:.2f} 秒")
        print("=" * 60)

    def check_system_info(self):
        """检查并打印系统信息"""
        print(f"\n🖥️  操作系统: {platform.system()} {platform.release()}")
        print(f"🖥️  系统架构: {platform.machine()}")
        print(f"✅ Python 版本: {sys.version.split(' ')[0]}")
        
        if self.verbose:
            print(f"📊 详细系统信息: {platform.platform()}")
            print(f"📊 Python 实现: {platform.python_implementation()}")

    def check_pytorch(self):
        """检查PyTorch安装状态"""
        try:
            print(f"\n📦 PyTorch 版本: {torch.__version__}")

            # 检查PyTorch的CUDA版本
            torch_cuda_version = (
                torch.version.cuda if hasattr(torch.version, "cuda") else "N/A"
            )
            print(f"🔗 PyTorch 内置CUDA版本: {torch_cuda_version}")

            # 检查CUDA是否可用
            self.cuda_available = torch.cuda.is_available()
            print(f"📊 CUDA 可用状态: {self.cuda_available}")

            # 尝试获取设备数量信息
            self._get_device_count()
            return True
        except ImportError:
            print("❌ 错误: 未安装PyTorch库")
            print("💡 提示: 请使用以下命令安装支持CUDA的PyTorch:")
            print(
                "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
            )
            return False
        except Exception as e:
            print(f"❌ 检查PyTorch过程中发生错误: {str(e)}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def _get_device_count(self):
        """获取GPU设备数量"""
        try:
            self.device_count = (
                torch.cuda.device_count() if hasattr(torch.cuda, "device_count") else 0
            )
            print(f"💻 检测到的GPU设备数量: {self.device_count}")
        except:
            print("💻 无法获取GPU设备数量")
            self.device_count = 0

    def check_gpu_devices(self):
        """检查并显示所有GPU设备的详细信息"""
        if not self.cuda_available or self.device_count == 0:
            return

        # 获取CUDA版本信息
        cuda_version = torch.version.cuda
        print(f"\n🎯 CUDA 版本: {cuda_version}")
        print(f"🎯 cuDNN 版本: {torch.backends.cudnn.version()}")
        print(f"💻 GPU 设备数量: {self.device_count}")

        # 遍历所有GPU设备并显示详细信息
        for i in range(self.device_count):
            self._show_gpu_details(i)
            self._test_gpu_computation(i)

        # 显示当前默认GPU设备
        current_device = torch.cuda.current_device()
        print(
            f"\n🎯 当前默认GPU设备: #{current_device} ({torch.cuda.get_device_name(current_device)})"
        )

    def _show_gpu_details(self, device_index):
        """显示指定GPU设备的详细信息"""
        try:
            device_name = torch.cuda.get_device_name(device_index)
            device_properties = torch.cuda.get_device_properties(device_index)

            print(f"\n--- GPU 设备 #{device_index+1} ---")
            print(f"设备名称: {device_name}")
            print(f"计算能力: {device_properties.major}.{device_properties.minor}")
            print(f"总内存: {device_properties.total_memory / 1024**3:.2f} GB")
            print(f"多处理器数量: {device_properties.multi_processor_count}")
            
            if self.verbose:
                # 安全地访问可能不存在的属性
                if hasattr(device_properties, 'memory_clock_rate'):
                    print(f"内存时钟频率: {device_properties.memory_clock_rate / 1e6:.2f} MHz")
                else:
                    print("内存时钟频率: 不支持的属性")
                    
                if hasattr(device_properties, 'clock_rate'):
                    print(f"核心时钟频率: {device_properties.clock_rate / 1e6:.2f} MHz")
                else:
                    print("核心时钟频率: 不支持的属性")
                    
                if hasattr(device_properties, 'max_threads_per_block'):
                    print(f"支持的最大线程数: {device_properties.max_threads_per_block}")
                else:
                    print("支持的最大线程数: 不支持的属性")
        except Exception as e:
            print(f"❌ 获取GPU #{device_index+1} 信息失败: {str(e)}")
            if self.verbose:
                import traceback
                traceback.print_exc()

    def _test_gpu_computation(self, device_index):
        """在指定GPU上执行简单计算测试"""
        try:
            # 创建一个简单的张量并移至GPU
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).to(f"cuda:{device_index}")
            result_tensor = test_tensor * 2
            
            if self.verbose:
                print(f"📊 计算测试输入: {test_tensor.cpu().numpy()}")
                print(f"📊 计算测试输出: {result_tensor.cpu().numpy()}")
                
            print(f"✅ GPU #{device_index+1} 计算测试成功")
        except Exception as e:
            print(f"❌ GPU #{device_index+1} 计算测试失败: {str(e)}")
            if self.verbose:
                import traceback
                traceback.print_exc()

    def check_nvidia_smi(self):
        """使用nvidia-smi命令检查NVIDIA设备状态"""
        print("\n🔧 执行系统命令检查NVIDIA设备和CUDA状态...")
        try:
            # 检查nvidia-smi命令是否存在
            if platform.system() == "Windows":
                nvidia_smi_output = subprocess.check_output(
                    ["nvidia-smi"], shell=True, stderr=subprocess.STDOUT
                ).decode("utf-8")
            else:
                # Linux/Mac
                nvidia_smi_output = subprocess.check_output(
                    ["nvidia-smi"], stderr=subprocess.STDOUT
                ).decode("utf-8")

            print("✅ 系统中检测到NVIDIA设备:")
            # 只显示前几行关键信息
            lines = nvidia_smi_output.split("\n")
            for line in lines[:7]:
                if line.strip():
                    print(f"   {line.strip()}")
            
            if self.verbose:
                print("\n📝 完整nvidia-smi输出:")
                for line in lines:
                    if line.strip():
                        print(f"   {line.strip()}")
        except Exception as e:
            print(f"⚠️  无法执行nvidia-smi命令: {str(e)}")
            if self.verbose:
                import traceback
                traceback.print_exc()

    def show_cuda_unavailable_reasons(self):
        """显示CUDA不可用的可能原因和解决方案"""
        print("\n❓ CUDA 不可用的可能原因:")
        print("1. 您的机器没有NVIDIA GPU")
        print("2. 未安装NVIDIA CUDA驱动程序")
        print("3. 安装的PyTorch版本不支持CUDA")
        print("4. CUDA驱动程序版本与PyTorch要求不兼容")
        print("5. 环境变量配置问题")

        print("\n💡 解决方案建议:")
        print("- 如果系统有NVIDIA GPU但PyTorch检测不到，请安装支持CUDA的PyTorch版本")
        print("- 在Windows上，推荐使用以下命令安装支持CUDA的PyTorch:")
        print(
            "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
        )
        print("- 在Linux上，推荐使用以下命令安装支持CUDA的PyTorch:")
        print(
            "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
        )
        print("- 请根据您的CUDA版本选择合适的PyTorch版本")

    def run_check(self):
        """运行完整的CUDA可用性检查"""
        try:
            self._print_header()
            self.check_system_info()

            pytorch_available = self.check_pytorch()
            if pytorch_available:
                if self.cuda_available:
                    self.check_gpu_devices()
                else:
                    self.check_nvidia_smi()
                    self.show_cuda_unavailable_reasons()
        except Exception as e:
            print(f"❌ 程序执行错误: {str(e)}")
            if self.verbose:
                import traceback
                traceback.print_exc()
        finally:
            self._print_footer()


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CUDA可用性检查器，用于诊断系统中的CUDA支持情况和GPU设备信息。",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""使用示例:
  基本检查:
    python cuda_checker.py
  
  详细检查:
    python cuda_checker.py -v"""
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="启用详细输出模式，显示更多调试信息"
    )
    
    return parser.parse_args()


def main():
    """主函数，执行CUDA检查流程"""
    args = parse_arguments()
    cuda_checker = CudaChecker(verbose=args.verbose)
    cuda_checker.run_check()


if __name__ == "__main__":
    main()
    