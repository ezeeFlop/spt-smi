from setuptools import setup, find_packages

setup(
    name="smi-client",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.24.0",
        "pydantic>=2.0.0",
    ],
    author="Sponge Theory",
    author_email="contact@sponge-theory.io",
    description="Python client for the Sponge Theory SMI API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://sponge-theory.ai",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
) 