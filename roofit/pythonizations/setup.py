import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="roofit_pythonizations",
    version="0.1",
    description="RooFit pythonizations",
    author="The ROOT developers",
    author_email="jonas.rembser@cern.ch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "python"},
    packages=setuptools.find_packages(where="python"),
    python_requires=">=3.8",
)
