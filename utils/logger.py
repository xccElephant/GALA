"""
统一日志工具模块

提供彩色日志输出，支持不同级别（成功/警告/错误/信息/调试）和格式化功能。
"""

import sys
from typing import Any, Optional


class Colors:
    """ANSI 颜色码"""
    # 基础颜色
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # 前景色
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # 亮色
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # 灰色（用于调试信息）
    GRAY = "\033[90m"
    
    @classmethod
    def disable(cls):
        """禁用颜色（用于非 TTY 环境）"""
        cls.RESET = ""
        cls.BOLD = ""
        cls.DIM = ""
        cls.BLACK = ""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.MAGENTA = ""
        cls.CYAN = ""
        cls.WHITE = ""
        cls.BRIGHT_RED = ""
        cls.BRIGHT_GREEN = ""
        cls.BRIGHT_YELLOW = ""
        cls.BRIGHT_BLUE = ""
        cls.BRIGHT_MAGENTA = ""
        cls.BRIGHT_CYAN = ""
        cls.BRIGHT_WHITE = ""
        cls.GRAY = ""


# 检测是否为 TTY，非 TTY 时禁用颜色
if not sys.stdout.isatty():
    Colors.disable()


class Logger:
    """
    统一日志类
    
    用法：
        log = Logger("Pipeline", verbose=True)
        log.info("Starting generation...")
        log.success("Placed at [1.88, 1.81, 0.46]")
        log.warning("No perfect placement found")
        log.error("Inference failed")
    """
    
    def __init__(self, module_name: str, verbose: bool = True):
        """
        初始化 Logger
        
        Args:
            module_name: 模块名称，用于日志前缀
            verbose: 是否打印日志，设为 False 可静默
        """
        self.module_name = module_name
        self.verbose = verbose
    
    def _format_prefix(self, color: str = "") -> str:
        """格式化前缀"""
        if color:
            return f"{color}[{self.module_name}]{Colors.RESET}"
        return f"[{self.module_name}]"
    
    def _print(self, message: str, color: str = "", prefix_color: str = ""):
        """内部打印方法"""
        if not self.verbose:
            return
        
        prefix = self._format_prefix(prefix_color)
        if color:
            print(f"{prefix} {color}{message}{Colors.RESET}")
        else:
            print(f"{prefix} {message}")
    
    def info(self, message: str):
        """打印信息日志（青色）"""
        self._print(message, color=Colors.BRIGHT_CYAN, prefix_color=Colors.BRIGHT_CYAN)
    
    def success(self, message: str):
        """打印成功日志（绿色，带 ✓）"""
        self._print(f"✓ {message}", color=Colors.BRIGHT_GREEN, prefix_color=Colors.BRIGHT_GREEN)
    
    def warning(self, message: str):
        """打印警告日志（黄色，带 ⚠）"""
        self._print(f"⚠ {message}", color=Colors.BRIGHT_YELLOW, prefix_color=Colors.BRIGHT_YELLOW)
    
    def error(self, message: str):
        """打印错误日志（红色，带 ✗）"""
        self._print(f"✗ {message}", color=Colors.BRIGHT_RED, prefix_color=Colors.BRIGHT_RED)
    
    def debug(self, message: str):
        """打印调试日志（灰色）"""
        self._print(message, color=Colors.GRAY, prefix_color=Colors.GRAY)
    
    def plain(self, message: str):
        """打印普通日志（无颜色）"""
        self._print(message)
    
    def step(self, current: int, total: int, description: str):
        """打印步骤进度"""
        progress = f"[{current}/{total}]"
        self._print(
            f"{Colors.BOLD}{progress}{Colors.RESET} {description}",
            prefix_color=Colors.BRIGHT_CYAN
        )
    
    def object_header(self, index: int, total: int, description: str):
        """打印物体处理标题"""
        if not self.verbose:
            return
        print(f"\n{Colors.BRIGHT_CYAN}[{self.module_name}]{Colors.RESET} "
              f"{Colors.BOLD}--- Object {index}/{total}: {description} ---{Colors.RESET}")
    
    def separator(self, char: str = "=", length: int = 60):
        """打印分隔线"""
        if not self.verbose:
            return
        print(char * length)
    
    def newline(self):
        """打印空行"""
        if not self.verbose:
            return
        print()
    
    def llm_block(self, block_type: str, content: str, model: str = None, provider: str = None):
        """
        打印 LLM 输入/输出块
        
        Args:
            block_type: "INPUT" 或 "OUTPUT"
            content: 内容
            model: 模型名称（可选）
            provider: 提供商（可选）
        """
        if not self.verbose:
            return
        
        self.separator()
        header = f"[Agent: {self.module_name}] LLM {block_type}"
        if model:
            header += f" (model={model}"
            if provider:
                header += f", provider={provider}"
            header += ")"
        print(f"{Colors.BRIGHT_MAGENTA}{header}{Colors.RESET}")
        self.separator()
        print(content)
        self.separator()
        print()
    
    def indent(self, message: str, level: int = 1):
        """打印缩进消息"""
        if not self.verbose:
            return
        indent_str = "  " * level
        print(f"{indent_str}{message}")
    
    def key_value(self, key: str, value: Any, indent_level: int = 1):
        """打印键值对（高亮值）"""
        if not self.verbose:
            return
        indent_str = "  " * indent_level
        print(f"{indent_str}{key}: {Colors.BOLD}{value}{Colors.RESET}")
    
    def placement_result(
        self,
        short_desc: str,
        center: list,
        rotation: list = None,
        inference_time: float = None,
        success: bool = True,
        partial: bool = False,
    ):
        """
        打印摆放结果（专用格式）
        
        Args:
            short_desc: 物体描述
            center: 中心坐标
            rotation: 旋转角度
            inference_time: 推理时间
            success: 是否成功
            partial: 是否为部分成功
        """
        if not self.verbose:
            return
        
        # 格式化坐标
        center_str = f"[{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]"
        
        # 构建消息
        msg_parts = [short_desc, "→", center_str]
        if rotation:
            rot_str = f"[{rotation[0]}, {rotation[1]}, {rotation[2]}]"
            msg_parts.append(f"rot={rot_str}")
        if inference_time is not None:
            msg_parts.append(f"({inference_time:.2f}s)")
        
        message = " ".join(msg_parts)
        
        if success:
            self.success(message)
        elif partial:
            self.warning(message)
        else:
            self.error(message)
    
    def validation_result(self, score: float, compliance: float, errors: list = None):
        """
        打印验证结果
        
        Args:
            score: 验证分数
            compliance: 指令符合度
            errors: 错误列表
        """
        if not self.verbose:
            return
        
        score_color = Colors.BRIGHT_GREEN if score >= 0.9 else (
            Colors.BRIGHT_YELLOW if score >= 0.5 else Colors.BRIGHT_RED
        )
        
        msg = f"score={score_color}{score:.2f}{Colors.RESET}, compliance={compliance:.1%}"
        if errors:
            msg += f", errors={errors}"
        
        self.indent(msg)
    
    def summary(self, **kwargs):
        """
        打印摘要信息
        
        用法：
            log.summary(
                successful=5,
                failed=1,
                elapsed_time="12.5s"
            )
        """
        if not self.verbose:
            return
        
        for key, value in kwargs.items():
            # 特殊处理某些键
            if key in ("successful", "success"):
                color = Colors.BRIGHT_GREEN
            elif key in ("failed", "error"):
                color = Colors.BRIGHT_RED
            elif key in ("warning", "partial"):
                color = Colors.BRIGHT_YELLOW
            else:
                color = Colors.BOLD
            
            key_display = key.replace("_", " ").capitalize()
            print(f"  {key_display}: {color}{value}{Colors.RESET}")


# 预定义的模块 Logger 实例（可直接导入使用）
def get_logger(module_name: str, verbose: bool = True) -> Logger:
    """
    获取 Logger 实例
    
    Args:
        module_name: 模块名称
        verbose: 是否打印日志
    
    Returns:
        Logger 实例
    """
    return Logger(module_name, verbose)
