from setuptools import setup, find_packages

setup(
    name='cyberllm',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'accelerate',
        'peft',
        'bitsandbytes',
        'transformers==4.30',
        'trl',
        "tensorboard",
        "scipy",
        "tensorflow"
    ],
    entry_points={
        'console_scripts': [
        ]
    }
)
