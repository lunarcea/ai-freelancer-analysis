from setuptools import setup, find_packages

setup(
    name="digital_labor_markets",
    version="0.1.0",
    description="Tools for analyzing skill metrics in digital labor markets",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "psutil>=5.8.0",
        "spacy>=3.1.0",
        "pytest>=6.2.5",
        "jupyter>=1.0.0",
        "statsmodels>=0.13.0",
        "pyarrow>=6.0.0",
        "openpyxl>=3.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)