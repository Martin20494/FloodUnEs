from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="floodunes",
    version='0.0.1',
    description="A package for estimating uncertainties in flood model outputs caused by square grid orientation and "
                "bathymetry estimation",
    author='M. Nguyen',
    author_email='martinnguyen20494@gmail.com',
    packages=find_packages(where="src"),
    include=["floodunes*"],
    python_requires=">=3.8"
)