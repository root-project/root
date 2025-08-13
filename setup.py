"""
setuptools-based build of ROOT.

This script uses setuptools API to steer a custom CMake build of ROOT. All build
configuration options are specified in the class responsible for building. A
custom extension module is injected in the setuptools setup to properly generate
the wheel with CPython extension metadata. Note that ROOT is first installed via
CMake into a temporary directory, then the ROOT installation artifacts are moved
to the final Python environment installation path, which often starts at
${ENV_PREFIX}/lib/pythonXX.YY/site-packages, before being packaged as a wheel.
"""

import os
import pathlib
import shlex
import subprocess
import tempfile

from setuptools import Extension, find_packages, setup
from setuptools.command.build import build as _build
from setuptools.command.install import install as _install

# Get the long description from the README file
SOURCE_DIR = pathlib.Path(__file__).parent.resolve()
LONG_DESCRIPTION = (SOURCE_DIR / "README.md").read_text(encoding="utf-8")

BUILD_DIR = tempfile.mkdtemp()
INSTALL_DIR = tempfile.mkdtemp()

# Name given to an internal directory within the build directory
# used to mimick the structure of the target installation directory
# in the user Python environment, usually named "site-packages"
ROOT_BUILD_INTERNAL_DIRNAME = "mock_site_packages"


class ROOTBuild(_build):
    def run(self):
        _build.run(self)

        # Configure ROOT build
        configure_command = shlex.split(
            "cmake "
            # gminimal=ON enables only a minimal set of components (cling+core+I/O+graphics)
            "-Dgminimal=ON -Dasimage=ON -Dopengl=OFF "
            "-Druntime_cxxmodules=ON -Dfail-on-missing=ON "  # Generic build configuration
            # Explicitly turned off components, even though they are already off because of gminimal, we want to keep
            # them listed here for documentation purposes:
            # - tmva-pymva, tpython: these components link against libPython, forbidden for manylinux compatibility,
            #   see https://peps.python.org/pep-0513/#libpythonx-y-so-1
            # - thisroot_scripts: the thisroot.* scripts are broken if CMAKE_INSTALL_PYTHONDIR!=CMAKE_INSTALL_LIBDIR
            "-Dtmva-pymva=OFF -Dtpython=OFF -Dthisroot_scripts=OFF "
            "-Dbuiltin_nlohmannjson=ON -Dbuiltin_tbb=ON -Dbuiltin_xrootd=ON "  # builtins
            "-Dbuiltin_lz4=ON -Dbuiltin_lzma=ON -Dbuiltin_zstd=ON -Dbuiltin_xxhash=ON "  # builtins
            "-Dpyroot=ON -Ddataframe=ON -Dxrootd=ON -Dssl=ON -Dimt=ON "
            "-Droofit=ON "
            # Next 4 paths represent the structure of the target binaries/headers/libs
            # as the target installation directory of the Python environment would expect
            f"-DCMAKE_INSTALL_BINDIR={ROOT_BUILD_INTERNAL_DIRNAME}/ROOT/bin "
            f"-DCMAKE_INSTALL_INCLUDEDIR={ROOT_BUILD_INTERNAL_DIRNAME}/ROOT/include "
            f"-DCMAKE_INSTALL_LIBDIR={ROOT_BUILD_INTERNAL_DIRNAME}/ROOT/lib "
            f"-DCMAKE_INSTALL_PYTHONDIR={ROOT_BUILD_INTERNAL_DIRNAME} "
            f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR} -B {BUILD_DIR} -S {SOURCE_DIR}"
        )
        subprocess.run(configure_command, check=True)

        # Run build with CMake
        build_command = f"cmake --build {BUILD_DIR} -j{os.cpu_count()}"
        subprocess.run(shlex.split(build_command), check=True)


class ROOTInstall(_install):
    def _get_install_path(self):
        if hasattr(self, "bdist_dir") and self.bdist_dir:
            install_path = self.bdist_dir
        else:
            install_path = self.install_lib

        return install_path

    def run(self):
        _install.run(self)

        install_cmd = f"cmake --build {BUILD_DIR} --target install"
        subprocess.run(shlex.split(install_cmd), check=True)

        install_path = self._get_install_path()

        # Copy ROOT installation tree to the ROOT package directory in the pip installation path
        self.copy_tree(os.path.join(INSTALL_DIR, ROOT_BUILD_INTERNAL_DIRNAME), install_path)

        root_package_dir = os.path.join(install_path, "ROOT")

        # After the copy of the "mock" package structure from the ROOT installations, these are the
        # leftover directories that still need to be copied
        self.copy_tree(os.path.join(INSTALL_DIR, "cmake"), os.path.join(root_package_dir, "cmake"))
        self.copy_tree(os.path.join(INSTALL_DIR, "etc"), os.path.join(root_package_dir, "etc"))
        self.copy_tree(os.path.join(INSTALL_DIR, "fonts"), os.path.join(root_package_dir, "fonts"))
        self.copy_tree(os.path.join(INSTALL_DIR, "icons"), os.path.join(root_package_dir, "icons"))
        self.copy_tree(os.path.join(INSTALL_DIR, "macros"), os.path.join(root_package_dir, "macros"))
        self.copy_tree(os.path.join(INSTALL_DIR, "man"), os.path.join(root_package_dir, "man"))
        self.copy_tree(os.path.join(INSTALL_DIR, "README"), os.path.join(root_package_dir, "README"))
        self.copy_tree(os.path.join(INSTALL_DIR, "tutorials"), os.path.join(root_package_dir, "tutorials"))
        self.copy_file(os.path.join(INSTALL_DIR, "LICENSE"), os.path.join(root_package_dir, "LICENSE"))

    def get_outputs(self):
        outputs = _install.get_outputs(self)
        return outputs


class DummyExtension(Extension):
    """
    Dummy CPython extension for setuptools setup.

    In order to generate the wheel with CPython extension metadata (i.e.
    producing one wheel per supported Python version), setuptools requires that
    at least one CPython extension is declared in the `ext_modules` kwarg passed
    to the `setup` function. Usually, declaring a CPython extension triggers
    compilation of the corresponding sources, but in this case we already do
    that in the CMake build step. This class defines a dummy extension that
    can be declared to setuptools while avoiding any further compilation step.
    """

    def __init__(_):
        super().__init__(name="Dummy", sources=[])


pkgs = find_packages("bindings/pyroot/pythonizations/python") + find_packages(
    "bindings/pyroot/cppyy/cppyy/python", include=["cppyy"]
)

s = setup(
    long_description=LONG_DESCRIPTION,
    package_dir={"": "bindings/pyroot/pythonizations/python", "cppyy": "bindings/pyroot/cppyy/cppyy/python"},
    packages=pkgs,
    # Crucial to signal this is not a pure Python package
    ext_modules=[DummyExtension()],
    cmdclass={"build": ROOTBuild, "install": ROOTInstall},
)
