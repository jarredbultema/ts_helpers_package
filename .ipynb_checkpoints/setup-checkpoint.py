import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open('Requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="datarobot_ts_helpers", 
    version="0.0.1dev3",
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
    install_requires = requirements #[
# #     'python3>=3.4',
#     'datarobot>=2.0',
#     'statsmodels>=0.10',
#     'plotly>=4.10',
#     'umap-learn>=0.3.10',
#     'ipywidgets>=7.2.1',
#     'scikit-learn>=0.22.1',  
#     'psycopg2>=2.8.5',
#     'matplotlib>=3.3',
#     'seaborn>=0.9'
#     ]
)