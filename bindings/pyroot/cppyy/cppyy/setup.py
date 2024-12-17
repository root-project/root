import codecs, glob, os, sys, re
from setuptools import setup, find_packages, Extension
from distutils import log

from setuptools.command.install import install as _install

add_pkg = ['cppyy', 'cppyy.__pyinstaller']
try:
    import __pypy__, sys
    version = sys.pypy_version_info
    requirements = ['cppyy-backend==1.15.3', 'cppyy-cling==6.32.8']
    if version[0] == 5:
        if version[1] <= 9:
            requirements = ['cppyy-backend<0.3', 'cppyy-cling<6.12']
            add_pkg += ['cppyy_compat']
        elif version[1] <= 10:
            requirements = ['cppyy-backend<0.4', 'cppyy-cling<=6.15']
    elif version[0] == 6:
        if version[1] <= 0:
            requirements = ['cppyy-backend<1.1', 'cppyy-cling<=6.15']
    elif version[0] == 7:
        if version[1] <= 3 and version[2] <= 3:
            requirements = ['cppyy-backend<=1.10', 'cppyy-cling<=6.18.2.3']
except ImportError:
    # CPython
    requirements = ['CPyCppyy==1.13.0', 'cppyy-backend==1.15.3', 'cppyy-cling==6.32.8']

setup_requirements = ['wheel']
if 'build' in sys.argv or 'install' in sys.argv:
    setup_requirements += requirements

here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# https://packaging.python.org/guides/single-sourcing-package-version/
def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


#
# customized commands
#
class my_install(_install):
    def __init__(self, *args, **kwds):
        if 0x3000000 <= sys.hexversion:
            super(_install, self).__init__(*args, **kwds)
        else:
          # b/c _install is a classobj, not type
            _install.__init__(self, *args, **kwds)

        try:
            import cppyy_backend as cpb
            self._pchname = 'allDict.cxx.pch.' + str(cpb.__version__)
        except (ImportError, AttributeError):
            self._pchname = None

    def run(self):
        # base install
        _install.run(self)

        # force build of the .pch underneath the cppyy package if not available yet
        install_path = os.path.join(os.getcwd(), self.install_libbase, 'cppyy')

        if self._pchname:
            try:
                import cppyy_backend as cpb
                if not os.path.exists(os.path.join(cpb.__file__, 'etc', self._pchname)):
                    log.info("installing pre-compiled header in %s", install_path)
                    cpb.loader.set_cling_compile_options(True)
                    cpb.loader.ensure_precompiled_header(install_path, self._pchname)
            except (ImportError, AttributeError):
                # ImportError may occur with wrong pip requirements resolution (unlikely)
                # AttributeError will occur with (older) PyPy as it relies on older backends
                self._pchname = None

    def get_outputs(self):
        outputs = _install.get_outputs(self)
        if self._pchname:
            # pre-emptively add allDict.cxx.pch, which may or may not be created; need full
            # path to make sure the final relative path is correct
            outputs.append(os.path.join(os.getcwd(), self.install_libbase, 'cppyy', self._pchname))
        return outputs


cmdclass = {
        'install': my_install }


setup(
    name='cppyy',
    version=find_version('python', 'cppyy', '_version.py'),
    description='Cling-based Python-C++ bindings',
    long_description=long_description,

    url='http://cppyy.readthedocs.org',

    # Author details
    author='Wim Lavrijsen',
    author_email='WLavrijsen@lbl.gov',

    license='LBNL BSD',

    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',

        'Topic :: Software Development',
        'Topic :: Software Development :: Interpreters',

        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: Implementation :: PyPy',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: C',
        'Programming Language :: C++',

        'Natural Language :: English'
    ],

    setup_requires=setup_requirements,
    install_requires=requirements,

    keywords='C++ bindings data science calling language integration',

    include_package_data=True,
    package_data={'': ['installer/cppyy_monkey_patch.py']},

    package_dir={'': 'python'},
    packages=find_packages('python', include=add_pkg),

    # TODO: numba_extensions will load all extensions even if the package
    # itself is not otherwise imported, just installed; in the case of cppyy,
    # that is currently too heavy (and breaks on conda)

    #entry_points={
    #    'numba_extensions': [
    #        'init = cppyy.numba_ext:_init_extension',
    #    ],
    #},

    cmdclass=cmdclass,

    zip_safe=False,
)
