from setuptools import setup, find_packages

setup(
    name="mcot-vg-accuracy",
    version="0.4.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
)
