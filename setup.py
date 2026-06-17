"""
项目设置脚本
用于在PyCharm中配置项目环境
"""

from setuptools import setup, find_packages

setup(
    name='yolov8-object-detection',
    version='1.0.0',
    description='YOLOv8目标检测 - 深度学习期末综合设计',
    author='',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'matplotlib>=3.7.0',
        'tqdm>=4.65.0',
        'Pillow>=10.0.0',
    ],
    python_requires='>=3.8',
)
