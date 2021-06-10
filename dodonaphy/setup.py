from pathlib import Path

from setuptools import setup, find_packages

setup(
    name='dodonaphy',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/koadman/dodonaphy',
    license='',
    author='Mathieu Fourment',
    author_email='mathieu.fourment@uts.edu.au',
    description='Hyperbolic embedding of phylogenies in pytorch',
    install_requires=[
        line.strip() for line in Path('requirements.txt').read_text('utf-8').splitlines()
    ]
)