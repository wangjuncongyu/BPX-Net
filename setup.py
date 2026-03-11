# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bpx-net",                 # 用户 pip install 的名字
    version="0.1.10",                # 你的版本号
    author="Jun Wang",              # 你的名字 (从代码注释中提取的)
    author_email="wjcy19870122@163.com", # 你的邮箱
    description="A Scikit-learn style wrapper for BPXNet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wangjuncongyu/BPX-Net", # 你的 GitHub 仓库地址
    packages=find_packages(),       # 自动发现 bpx_net 目录及其子包
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.7',
    install_requires=[              # 核心依赖！pip 安装你的包时会自动下载这些
        "torch==2.9.1",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scikit-learn==1.1.0",
    ],
)