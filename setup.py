from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mmm-validation-dataset",
    version="1.0.0",
    author="Annabel Castro",
    author_email="annabelcasttro@gmail.com",
    description="Synthetic MMM dataset with ground truth for validating attribution decomposition methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/annabelcasttro/marketing-mix-modeling-research",
    py_modules=["synthetic_mmm_dataset"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
    ],
)
