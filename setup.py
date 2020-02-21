"""Setup.py for Landscape
See: https://github.com/ynshen/Landscape
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='seq-landscape',   # this is the name registered on pypi
    version='0.1.0a1',
    description='Analysis on sequencing landscape',
    long_description=long_description,
    long_description_content_type='text/markdown', # to use markdown, make sure wheel, setuptools, twine are up to date
    url='https://github.com/ynshen/Landscape',
    author='Yuning Shen',
    author_email='ynshen23@gmail.com',
    keywords='',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.5',
    install_requires=[
        'pandas',
        'numpy',
        'scipy'
        'matplotlib'
    ],
    classifiers=[
        'OSI Approved :: MIT License'
    ],
    project_urls={
        'Bug Reports': 'https://github.com/ynshen/Landscape/issues/',
        'Source': 'https://github.com/ynshen/Landscape',
    },
)
