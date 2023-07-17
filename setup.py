import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="gzip-classifier",
    version="1.0",
    author="Brian Schrader",
    author_email="brian@brianschrader.com",
    description="A gzip-based text-classification system.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sonictherocketman/gzip-classifier",
    project_urls={
        "Bug Tracker": "https://github.com/Sonictherocketman/gzip-classifier/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
)
