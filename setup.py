from setuptools import setup, find_packages

setup(
    name="ascredit",
    version="0.1.0",
    description="Avellaneda-Stoikov optimal market-making adapted to credit (spread space)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["numpy>=1.24", "matplotlib>=3.7"],
    license="MIT",
)
