import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open('Requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="datarobot_ts_helpers", 
    version="0.0.1beta1",
    author="Jarred Bultema",
    author_email="jarred.bultema@datarobot.com",
    description="A package with helper scripts for complex DataRobot AutoTS use cases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jarredbultema/ts_helpers_package",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.4',
    install_requires = requirements 
)