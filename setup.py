from pathlib import Path

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name="dodonaphy",
    version="0.0.1",
    packages=find_packages(),
    url="https://github.com/koadman/dodonaphy",
    license="",
    author="Mathieu Fourment",
    author_email="mathieu.fourment@uts.edu.au",
    description="Hyperbolic embedding of phylogenies in pytorch",
    install_requires=[
        line.strip()
        for line in Path("requirements.txt").read_text("utf-8").splitlines()
    ],
    ext_modules=cythonize(
        [
            Extension(
                "dodonaphy.Cutils", ["dodonaphy/cython/Cutils.pyx"], include_dirs=[np.get_include()]
            ),
            Extension(
                "dodonaphy.Cphylo", ["dodonaphy/cython/Cphylo.pyx"], include_dirs=[np.get_include()]
            ),
            Extension(
                "dodonaphy.Chyp_torch", ["dodonaphy/cython/Chyp_torch.pyx"], include_dirs=[np.get_include()]
            ),
            Extension(
                "dodonaphy.Chyp_np", ["dodonaphy/cython/Chyp_np.pyx"], include_dirs=[np.get_include()]
            ),
            Extension(
                "dodonaphy.Cpeeler", ["dodonaphy/cython/Cpeeler.pyx"], include_dirs=[np.get_include()]
            ),
            Extension(
                "dodonaphy.Ctransforms", ["dodonaphy/cython/Ctransforms.pyx"], include_dirs=[np.get_include()]
            ),
        ],
        annotate=True
    ),
    entry_points={"console_scripts": ["dodo = dodonaphy.run:main"]},
)
