from setuptools import setup, find_packages

setup(
    name="market-intelligence",
    version="1.0.0",
    description="Real-time market intelligence system for Indian stock market analysis",
    author="Market Intelligence Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        line.strip() 
        for line in open("requirements.txt").readlines() 
        if line.strip() and not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "market-intel=src.main:main",
        ],
    },
)
