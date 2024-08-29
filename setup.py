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
    download_url='https://github.com/StanCDev/DimRed/archive/v_01.tar.gz',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache 2.0 license",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)