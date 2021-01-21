from setuptools import find_packages, setup

with open("../README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='datastack',
    version='0.0.6',
    author='Max Luebbering',
    description="DataStack, a stream based solution for machine learning dataset retrieval and storage",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/le1nux/datastack",
    packages=find_packages(),
    install_requires=[
                      "pytest",
                      "pytest-cov",
                      "torch",
                      "torchvision",
                      "tqdm",
                      "PyYAML"
                      ],
    python_requires=">=3.7"
)
