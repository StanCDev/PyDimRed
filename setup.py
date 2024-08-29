import setuptools

# Load the long_description from README.md
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DimRed",
    version="0.0.1",
    author="StanCDev",
    author_email="stancastellana@icloud.com",
    description="Dimensionality reduction library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StanCDev/DimRed",
    download_url='https://github.com/StanCDev/DimRed/archive/refs/tags/v_01.tar.gz',
    install_requires= 
    [
        "sklearn",
        "numpy",
        "pandas",
        "joblib",
        "seaborn",
        "matplotlib",
        "umap",
        "trimap",
        "pacmap",
        "openTSNE",
        "pathlib",
    ],
    # packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache-2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)