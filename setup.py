import setuptools

# Load the long_description from README.md
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyDimRed",
    version="0.0.2",
    author="StanCDev",
    author_email="stancastellana@icloud.com",
    description="Dimensionality reduction library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StanCDev/PyDimRed",
    download_url='https://github.com/StanCDev/PyDimRed/archive/refs/tags/v_02.tar.gz',
    packages= setuptools.find_packages(),
    install_requires= 
    [
        "scikit-learn",
        "numpy",
        "pandas",
        "joblib",
        "seaborn",
        "matplotlib",
        "umap-learn",
        "trimap",
        "pacmap",
        "openTSNE",
        "pathlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)