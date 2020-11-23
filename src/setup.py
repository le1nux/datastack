from setuptools import find_packages, setup

setup(
    name='data_stack',
    version='0.0.1',
    author='Max Luebbering',
    description="DataStack, a stream based solution for machine learning dataset retrieval and storage",
    url="https://github.com/le1nux/datastack",
    packages=find_packages(),
    install_requires=["torch",
                      "torchvision",
                      "tqdm",
                      "PyYAML"
                      ],
    python_requires=">=3.7"
)
