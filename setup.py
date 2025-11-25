from setuptools import setup, find_packages

setup(
    name="semg_preprocessing",
    version="0.1.0",
    description="Surface Electromyography (sEMG) signal preprocessing toolkit",
    author="PRIMOCOSMOS",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "ruptures>=1.1.7",
        "matplotlib>=3.4.0",
    ],
    python_requires=">=3.7",
)
