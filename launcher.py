"""
YAOSU 一键启动器
双击即可启动 Streamlit 服务并自动打开浏览器
"""
import subprocess
import webbrowser
import time
import os
import sys
import socket
import shutil


def is_port_in_use(port: int) -> bool:
    """检查指定端口是否已被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_python():
    """查找系统 Python 解释器"""
    # 1. 开发模式：直接用当前解释器
    if not getattr(sys, "frozen", False):
        return sys.executable

    # 2. 从 PATH 中找（PyInstaller 打包后）
    for name in ["python", "python3"]:
        found = shutil.which(name)
        if found:
            return found

    # 3. Windows 注册表查找
    if sys.platform == "win32":
        try:
            import winreg
            for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for subkey in (
                    r"Software\Python\PythonCore\3.13\InstallPath",
                    r"Software\Python\PythonCore\3.12\InstallPath",
                    r"Software\Python\PythonCore\3.11\InstallPath",
                    r"Software\Python\PythonCore\3.10\InstallPath",
                    r"Software\Python\PythonCore\3.9\InstallPath",
                ):
                    try:
                        with winreg.OpenKey(root, subkey) as key:
                            path = winreg.QueryValue(key, "")
                            if path:
                                python = os.path.join(path, "python.exe")
                                if os.path.exists(python):
                                    return python
                    except OSError:
                        continue
        except Exception:
            pass

    # 4. 常见安装路径兜底
    for ver in ("313", "312", "311", "310", "39"):
        candidates = [
            os.path.expandvars(fr"%LOCALAPPDATA%\Programs\Python\Python{ver}\python.exe"),
            fr"C:\Python{ver}\python.exe",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path

    return None


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    port = 8501
    url = f"http://localhost:{port}"

    if is_port_in_use(port):
        print(f"端口 {port} 已被占用，可能服务已在运行，直接打开浏览器。")
        webbrowser.open(url)
        input("按回车键退出...")
        return

    python_exe = find_python()
    if python_exe is None:
        print("错误: 未找到 Python，请确认已安装 Python 3.9+ 并添加到 PATH。")
        input("按回车键退出...")
        sys.exit(1)

    print(f"使用 Python: {python_exe}")
    print("正在启动服务，请稍候...")

    if sys.platform == "win32":
        process = subprocess.Popen(
            [python_exe, "-m", "streamlit", "run", "app.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        process = subprocess.Popen(
            [python_exe, "-m", "streamlit", "run", "app.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # 等待端口就绪，最多等 30 秒
    for i in range(30):
        time.sleep(1)
        if is_port_in_use(port):
            print(f"服务已启动！正在打开浏览器: {url}")
            webbrowser.open(url)
            print("\n按 Ctrl+C 可停止服务。")
            input("按回车键退出本窗口（不影响服务运行）...")
            return

    print("警告: 服务启动超时。请手动运行: streamlit run app.py")
    print(f"然后访问: {url}")
    input("按回车键退出...")


if __name__ == "__main__":
    main()
