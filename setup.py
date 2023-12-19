from setuptools import setup, find_packages

with open("README.rst", "r", encoding="utf-8") as f:
    __long_description__ = f.read()
  
setup(
    name='Wenlin Li',
    version='0.1',
    packages=find_packages(),
    install_requires=[
    ],
    packages = ["SpaCEX"],
    description='Gene embeddings geneartion, Enhancement of the transcriptomic coverage, SVG detection, Spatial clustering',
    author='Wenlin Li',
    author_email='zipging@gmail.com',
    url='https://github.com/WLatSunLab/SpaCEX',
)
