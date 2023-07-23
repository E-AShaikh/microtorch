from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='microtorch',
    packages=find_packages(),
    version='0.1.0',
    description='A tiny vector-valued autograd engine with a small PyTorch-like neural network library on top.',
    author='Ezzdeen A.Shaikh',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/E-AShaikh/microtorch',
    license='MIT',
)