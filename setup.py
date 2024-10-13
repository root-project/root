"""
setuptools-based build of ROOT.

This script uses setuptools API to steer a custom CMake build of ROOT. All build
configuration options are specified in the class responsible for building. The
installation step produces the following output target install directory:

```
site-packages/
-- cppyy/
-- cppyy_backend/
-- libcppyy.so
-- libcppyy_backend.so
-- libROOTPythonizations.so
-- ROOT/
```

A custom extension module is injected in the setuptools setup to properly
generate the wheel with CPython extension metadata.
"""

from setuptools import setup, find_packages, Extension
import pathlib
import tempfile
from setuptools.command.build import build as _build
from setuptools.command.install import install as _install

import subprocess
import os
import shlex

# Get the long description from the README file
SOURCE_DIR = pathlib.Path(__file__).parent.resolve()
LONG_DESCRIPTION = (SOURCE_DIR / "README.md").read_text(encoding="utf-8")

BUILD_DIR = tempfile.mkdtemp()
INSTALL_DIR = tempfile.mkdtemp()


class ROOTBuild(_build):
    def run(self):
        _build.run(self)

        # Configure ROOT build
        base_opts = shlex.split("cmake -GNinja")
        mode_opts = shlex.split(
            "-Dbuiltin_nlohmannjson=ON -Dbuiltin_tbb=ON -Dbuiltin_xrootd=ON "  # builtins
            "-Dbuiltin_lz4=ON -Dbuiltin_lzma=ON -Dbuiltin_zstd=ON -Dbuiltin_xxhash=ON"  # builtins
            "-Druntime_cxxmodules=ON -Drpath=ON -Dfail-on-missing=ON "  # Generic build configuration
            "-Dgminimal=ON -Dasimage=ON -Dopengl=OFF "  # Graphics
            "-Dpyroot=ON -Ddataframe=ON -Dxrootd=ON -Dimt=ON "
            "-Droofit=ON")
        dirs_opts = shlex.split(
            f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR} -B {BUILD_DIR} -S {SOURCE_DIR}")
        configure_command = base_opts + mode_opts + dirs_opts
        subprocess.run(configure_command, check=True)

        # Run build with CMake
        build_command = f"cmake --build {BUILD_DIR} -j{os.cpu_count()}"
        subprocess.run(shlex.split(build_command), check=True)


class ROOTInstall(_install):
    def _get_install_path(self):
        if hasattr(self, 'bdist_dir') and self.bdist_dir:
            install_path = self.bdist_dir
        else:
            install_path = self.install_lib

        return install_path

    def run(self):
        _install.run(self)

        install_cmd = f"cmake --build {BUILD_DIR} --target install"
        subprocess.run(shlex.split(install_cmd), check=True)

        install_path = self._get_install_path()

        lib_dir = os.path.join(INSTALL_DIR, "lib")

        # Copy ROOT installation tree to the ROOT package directory in the pip installation path
        self.copy_tree(INSTALL_DIR, os.path.join(install_path, "ROOT"))

        # Copy cppyy packages separately
        self.copy_tree(os.path.join(lib_dir, "cppyy"),
                       os.path.join(install_path, "cppyy"))
        self.copy_tree(os.path.join(lib_dir, "cppyy_backend"),
                       os.path.join(install_path, "cppyy_backend"))

        # Finally copy CPython extension libraries
        extlibs = ["libcppyy.so", "libcppyy_backend.so",
                   "libROOTPythonizations.so"]
        for ext in extlibs:
            self.copy_file(os.path.join(lib_dir, ext), install_path)

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


pkgs = (
    find_packages('bindings/pyroot/pythonizations/python') +
    find_packages('bindings/pyroot/cppyy/cppyy/python', include=['cppyy'])
)

s = setup(
    long_description=LONG_DESCRIPTION,
    package_dir={'': 'bindings/pyroot/pythonizations/python',
                 'cppyy': 'bindings/pyroot/cppyy/cppyy/python'},
    packages=pkgs,
    # Crucial to signal this is not a pure Python package
    ext_modules=[DummyExtension()],
    cmdclass={
        'build': ROOTBuild,
        'install': ROOTInstall},
)
