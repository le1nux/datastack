from setuptools import find_packages, setup

with open("../README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='datastack',
    version='0.0.1',
    author='Max Luebbering',
    description="DataStack, a stream based solution for machine learning dataset retrieval and storage",
    long_description=long_description,
    url="https://github.com/le1nux/datastack",
    packages=find_packages(),
    install_requires=["torch",
                      "torchvision",
                      "tqdm",
                      "PyYAML"
                      ],
    python_requires=">=3.7"
)
