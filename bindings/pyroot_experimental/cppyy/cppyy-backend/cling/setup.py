import os, sys, subprocess
import multiprocessing
from setuptools import setup, find_packages
from distutils import log
from distutils.command.build import build as _build
from distutils.command.clean import clean as _clean
from distutils.dir_util import remove_tree
from setuptools.command.install import install as _install
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
from distutils.errors import DistutilsSetupError
from codecs import open

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

builddir = None
def get_builddir():
    """cppyy_backend build."""
    global builddir
    if builddir is None:
        topdir = os.getcwd()
        builddir = os.path.join(topdir, 'builddir')
    return builddir

srcdir = None
def get_srcdir():
    """cppyy_backend source."""
    global srcdir
    if srcdir is None:
        topdir = os.getcwd()
        srcdir = os.path.join(topdir, 'src')
    return srcdir

prefix = None
def get_prefix():
    """cppyy_backend installation."""
    global prefix
    if prefix is None:
        prefix = os.path.join(get_builddir(), 'install', 'cppyy_backend')
    return prefix


class my_cmake_build(_build):
    def run(self):
        # base run
        _build.run(self)

        # custom run
        log.info('Now building cppyy_backend')
        builddir = get_builddir()
        prefix   = get_prefix()
        srcdir   = get_srcdir()
        if not os.path.exists(srcdir):
            log.info('No src directory ... creating with "python create_src_directory.py"')
            if subprocess.call(['python', 'create_src_directory.py']) != 0:
                log.error('ERROR: the source directory "%s" does not exist' % srcdir)
                log.error('Please run "python create_src_directory.py" first.')
                sys.exit(1)

        if not os.path.exists(builddir):
            log.info('Creating build directory %s ...' % builddir)
            os.makedirs(builddir)

        # get C++ standard to use, if set
        try:
            stdcxx = os.environ['STDCXX']
        except KeyError:
            stdcxx = '14'

        if not stdcxx in ['11', '14', '17']:
            log.fatal('FATAL: envar STDCXX should be one of 11, 14, or 17')
            sys.exit(1)

        stdcxx='-Dcxx'+stdcxx+'=ON'

        log.info('Running cmake for cppyy_backend')
        if subprocess.call([
                'cmake', srcdir, stdcxx,
                '-Dminimal=ON', '-Dasimage=OFF', '-Droot7=OFF', '-Dhttp=OFF', '-Dbuiltin_freetype=OFF',
                '-DCMAKE_BUILD_TYPE=RelWithDebInfo',
                '-DCMAKE_INSTALL_PREFIX='+prefix], cwd=builddir) != 0:
            raise DistutilsSetupError('Failed to configure cppyy_backend')

        # default to using all available cores (x2 if hyperthreading enabled)
        nprocs = os.getenv("MAKE_NPROCS") or '0'
        try:
            nprocs = int(nprocs)
        except ValueError:
            log.warn("Integer expected for MAKE_NPROCS, but got %s (ignored)", nprocs)
            nprocs = 0
        if nprocs < 1:
            nprocs = multiprocessing.cpu_count()
        build_args = ['--build', '.', '--config', 'RelWithDebInfo', '--']
        if 'win32' in sys.platform:
            build_args.append('/maxcpucount:' + str(nprocs))
        else:
            build_args.append('-j' + str(nprocs))
        log.info('Now building cppyy_backend and dependencies ...')
        if subprocess.call(['cmake'] + build_args, cwd=builddir) != 0:
            raise DistutilsSetupError('Failed to build cppyy_backend')

        log.info('Build finished')

class my_clean(_clean):
    def run(self):
        # Custom clean. Clean everything except that which the base clean
        # (see below) or create_src_directory.py is responsible for.
        topdir = os.getcwd()
        if self.all:
            # remove build directories
            for directory in (get_builddir(),
                              os.path.join(topdir, "python", "cppyy_cling.egg-info")):
                if os.path.exists(directory):
                    remove_tree(directory, dry_run=self.dry_run)
                else:
                    log.warn("'%s' does not exist -- can't clean it",
                             directory)
        # Base clean.
        _clean.run(self)

class my_install(_install):
    def _get_install_path(self):
        # depending on goal, copy over pre-installed tree
        if hasattr(self, 'bdist_dir') and self.bdist_dir:
            install_path = self.bdist_dir
        else:
            install_path = self.install_lib
        return install_path

    def run(self):
        # base install
        _install.run(self)

        # custom install of backend
        log.info('Now installing cppyy_backend')
        builddir = get_builddir()
        if not os.path.exists(builddir):
            raise DistutilsSetupError('Failed to find build dir!')

        prefix = get_prefix()
        log.info('Now creating installation under %s ...', prefix)
        install_args = ['--build', '.', '--config', 'RelWithDebInfo', '--target', 'install']
        if subprocess.call(['cmake'] + install_args, cwd=builddir) != 0:
            raise DistutilsSetupError('Failed to install cppyy_backend')

        prefix_base = os.path.join(get_prefix(), os.path.pardir)
        install_path = self._get_install_path()
        log.info('Copying installation to: %s ...', install_path)
        self.copy_tree(prefix_base, install_path)

        log.info('Install finished')

    def get_outputs(self):
        outputs = _install.get_outputs(self)
        outputs.append(os.path.join(self._get_install_path(), 'cppyy_backend'))
        return outputs

class my_bdist_wheel(_bdist_wheel):
    def finalize_options(self):
     # this is a universal, but platform-specific package; a combination
     # that wheel does not recognize, thus simply fool it
        from distutils.util import get_platform
        self.plat_name = get_platform()
        _bdist_wheel.finalize_options(self)
        self.root_is_pure = True


setup(
    name='cppyy-cling',
    description='Re-packaged Cling, as backend for cppyy',
    long_description=long_description,
    url='https://root.cern.ch/cling',

    # Author details
    author='ROOT Developers',
    author_email='rootdev@cern.ch',

    version='6.12.6.2',

    license='LLVM: UoI-NCSA; ROOT: LGPL 2.1',

    classifiers=[
        'Development Status :: 6 - Mature',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        'Topic :: Software Development',
        'Topic :: Software Development :: Interpreters',

        'License :: OSI Approved :: University of Illinois/NCSA Open Source License',
        'License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)',

        'Operating System :: POSIX',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: C',
        'Programming Language :: C++',

        'Natural Language :: English'
    ],

    keywords='interpreter development',

    include_package_data=True,
    package_data={'': ['cmake/*.cmake']},

    package_dir={'': 'python'},
    packages=find_packages('python', include=['cppyy_backend']),

    cmdclass = {
        'build': my_cmake_build,
        'clean': my_clean,
        'install': my_install,
        'bdist_wheel': my_bdist_wheel,
    },

    entry_points={
        "console_scripts": [
            "cling-config = cppyy_backend._cling_config:main",
            "genreflex = cppyy_backend._genreflex:main",
            "rootcling = cppyy_backend._rootcling:main",
            "cppyy-generator = cppyy_backend._cppyy_generator:main",
        ],
    },
)
