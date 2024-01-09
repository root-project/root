"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib
from setuptools.command.build import build as _build
from setuptools.command.install import install as _install

import subprocess
import os
import glob
import shlex
here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

SOURCE_DIR = os.path.expanduser("~/Programs/rootproject/rootsrc")
BUILD_DIR = os.path.expanduser("~/Programs/rootproject/pip-tests/build")
INSTALL_DIR = os.path.expanduser("~/Programs/rootproject/pip-tests/install")


class my_cmake_build(_build):
    def run(self):
        # base run
        _build.run(self)

        if not os.path.exists(SOURCE_DIR):
            raise RuntimeError("Could not find the source directory!")

        if not os.path.exists(BUILD_DIR):
            os.makedirs(BUILD_DIR)

        if not os.path.exists(INSTALL_DIR):
            os.makedirs(INSTALL_DIR)

        base_opts = shlex.split("cmake -GNinja -Dccache=ON")
        mode_opts = shlex.split("-Dminimal=ON -Dpyroot=ON -Druntime_cxxmodules=ON")
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
        install_libdir = os.path.join(install_path, "ROOT", "lib")

        self.copy_tree(os.path.join(INSTALL_DIR, "lib"), os.path.join(install_path, "ROOT", "lib"))
        self.copy_tree(os.path.join(INSTALL_DIR, "include"), os.path.join(install_path, "ROOT", "include"))
        self.copy_tree(os.path.join(INSTALL_DIR, "bin"), os.path.join(install_path, "ROOT", "bin"))
        self.copy_tree(os.path.join(INSTALL_DIR, "etc"), os.path.join(install_path, "ROOT", "etc"))

        self.copy_tree(os.path.join(lib_dir, "cppyy"),
                       os.path.join(install_path, "cppyy"))
        self.copy_tree(os.path.join(lib_dir, "cppyy_backend"),
                       os.path.join(install_path, "cppyy_backend"))

        extlibs = ["libcppyy3_11.so", "libcppyy_backend3_11.so", "libROOTPythonizations3_11.so"]
        for ext in extlibs:
            self.copy_file(os.path.join(lib_dir, ext), install_path)

        cppextensions = [os.path.join(install_path, ext) for ext in extlibs]
        for ext in cppextensions:
            subprocess.run(shlex.split(f"patchelf --set-rpath \$ORIGIN:\$ORIGIN/ROOT/lib {ext}"))

        rootlibs = glob.glob(os.path.join(install_libdir, "lib*.so"))
        for lib in rootlibs:
            subprocess.run(shlex.split(f"patchelf --set-rpath \$ORIGIN {lib}"))

    def get_outputs(self):
        outputs = _install.get_outputs(self)
        return outputs


pkgs = (
    find_packages('bindings/pyroot/pythonizations/python') +
    find_packages('bindings/pyroot/cppyy/cppyy/python', include=['cppyy'])
)

s = setup(

    package_dir={'': 'bindings/pyroot/pythonizations/python',
                 'cppyy': 'bindings/pyroot/cppyy/cppyy/python'},
    packages=pkgs,

    cmdclass={
        'build': my_cmake_build,
        'install': my_install},

)
