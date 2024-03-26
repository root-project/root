"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages, Extension
import pathlib
import shutil
import tempfile
from setuptools.command.build import build as _build
from setuptools.command.install import install as _install

import subprocess
import os
import shlex
import sys
here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

SOURCE_DIR = os.getcwd()
BUILD_DIR = tempfile.mkdtemp()
INSTALL_DIR = tempfile.mkdtemp()

# SOURCE_DIR = os.path.join(os.sep, *"host/home/vpadulan/Programs/rootproject/rootsrc".split("/"))
# BUILD_DIR = os.path.join(os.sep, *"host/home/vpadulan/Programs/rootproject/pip-tests/build".split("/"))
# INSTALL_DIR = os.path.join(os.sep, *"host/home/vpadulan/Programs/rootproject/pip-tests/install".split("/"))

class my_cmake_build(_build):
    def run(self):
        # base run
        _build.run(self)

        # if not os.path.exists(SOURCE_DIR):
            # subprocess.run(shlex.split("git clone https://github.com/root-project/root.git rootsrc"), check=True)

        if not os.path.exists(BUILD_DIR):
            os.makedirs(BUILD_DIR)

        if not os.path.exists(INSTALL_DIR):
            os.makedirs(INSTALL_DIR)

        base_opts = shlex.split("cmake -GNinja -Dccache=ON")
        mode_opts = shlex.split(
            "-Dgminimal=ON -Dasimage=ON -Drpath=ON -Dpyroot=ON "
            "-Druntime_cxxmodules=ON -Ddataframe=ON -Dxrootd=ON -Dimt=ON "
            "-Droofit=ON")
        dirs_opts = shlex.split(
            f"-DCMAKE_INSTALL_PREFIX={INSTALL_DIR} -B {BUILD_DIR} -S {SOURCE_DIR}")
        configure_command = base_opts + mode_opts + dirs_opts
        subprocess.run(configure_command, check=True)

        build_command = f"cmake --build {BUILD_DIR} -j{os.cpu_count()}"
        subprocess.run(shlex.split(build_command), check=True)



class my_install(_install):
    def _get_install_path(self):
        # depending on goal, copy over pre-installed tree
        if hasattr(self, 'bdist_dir') and self.bdist_dir:
            install_path = self.bdist_dir
        else:
            install_path = self.install_lib

        print(f"\n\n{install_path=}\n\n")
        print(f"\n\n{self.install_usersite=}\n\n")
        print(f"\n\n{self.install_userbase=}\n\n")

        return install_path

    def run(self):
        # base install
        _install.run(self)

        install_cmd = f"cmake --build {BUILD_DIR} --target install"
        subprocess.run(shlex.split(install_cmd), check=True)

        install_path = self._get_install_path()

        lib_dir = os.path.join(INSTALL_DIR, "lib")

        # Copy ROOT installation tree to the ROOT package directory in the pip installation path
        self.copy_tree(INSTALL_DIR, os.path.join(install_path, "ROOT"))
        # self.copy_tree(os.path.join(INSTALL_DIR, "lib"), os.path.join(install_path, "ROOT", "lib"))
        # self.copy_tree(os.path.join(INSTALL_DIR, "include"), os.path.join(install_path, "ROOT", "include"))
        # self.copy_tree(os.path.join(INSTALL_DIR, "bin"), os.path.join(install_path, "ROOT", "bin"))
        # self.copy_tree(os.path.join(INSTALL_DIR, "etc"), os.path.join(install_path, "ROOT", "etc"))

        self.copy_tree(os.path.join(lib_dir, "cppyy"),
                       os.path.join(install_path, "cppyy"))
        self.copy_tree(os.path.join(lib_dir, "cppyy_backend"),
                       os.path.join(install_path, "cppyy_backend"))

        extlibs = ["libcppyy.so", "libcppyy_backend.so", "libROOTPythonizations.so"]
        for ext in extlibs:
            self.copy_file(os.path.join(lib_dir, ext), install_path)

    def get_outputs(self):
        outputs = _install.get_outputs(self)
        return outputs


class CMakeExtension(Extension):
    """
    An extension to run the cmake build

    This simply overrides the base extension class so that setuptools
    doesn't try to build your sources for you
    """

    def __init__(self, name, sources=[]):
        super().__init__(name = name, sources = sources)

pkgs = (
    find_packages('bindings/pyroot/pythonizations/python') +
    find_packages('bindings/pyroot/cppyy/cppyy/python', include=['cppyy'])
)

s = setup(

    package_dir={'': 'bindings/pyroot/pythonizations/python',
                 'cppyy': 'bindings/pyroot/cppyy/cppyy/python'},
    packages=pkgs,
    ext_modules=[
        CMakeExtension(name="pippo"),
    ],

    cmdclass={
        'build': my_cmake_build,
        'install': my_install},

)


# # Cleanup temporary directories
# shutil.rmtree(BUILD_DIR)
# shutil.rmtree(INSTALL_DIR)
