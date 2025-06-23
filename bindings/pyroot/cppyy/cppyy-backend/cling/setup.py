import codecs, multiprocessing, os, sys, subprocess, stat, re
from setuptools import setup, find_packages
from distutils import log

from setuptools.dist import Distribution
from distutils.command.build import build as _build
from distutils.command.clean import clean as _clean
from distutils.dir_util import remove_tree
from setuptools.command.install import install as _install
try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
    has_wheel = True
except ImportError:
    has_wheel = False
from distutils.errors import DistutilsSetupError


requirements = []
setup_requirements = ['wheel']+requirements

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
# platform-dependent helpers
#
_is_manylinux = None
def is_manylinux():
    global _is_manylinux
    if _is_manylinux is None:
        _is_manylinux = False
        try:
            for line in open('/etc/redhat-release').readlines():
              # mark manylinux1, manylinux2010, or manylinux2014
                if 'CentOS release 5.11 (Final)' in line or \
                   'CentOS release 6.10 (Final)' in line or \
                   'CentOS Linux release 7.9.2009 (Core)' in line:
                    _is_manylinux = True
                    break
        except (OSError, IOError):
            pass
    return _is_manylinux

def get_build_type():
    if is_manylinux() or 'win32' in sys.platform:
        # debug info too large for wheels
        return 'Release'
    return 'RelWithDebInfo'

builddir = None
def get_builddir():
    """cppyy-cling build."""
    global builddir
    if builddir is None:
        topdir = os.getcwd()
        builddir = os.path.join(topdir, 'builddir')
    return builddir

srcdir = None
def get_srcdir():
    """cppyy-cling source."""
    global srcdir
    if srcdir is None:
        topdir = os.getcwd()
        srcdir = os.path.join(topdir, 'src')
    return srcdir

prefix = None
def get_prefix():
    """cppyy-cling installation."""
    global prefix
    if prefix is None:
        prefix = os.path.join(get_builddir(), 'install', 'cppyy_backend')
    return prefix


#
# customized commands
#
class my_cmake_build(_build):
    def run(self):
        # base run
        _build.run(self)

        # custom run
        log.info('Now building cppyy-cling')
        builddir = get_builddir()
        prefix   = get_prefix()
        srcdir   = get_srcdir()
        if not os.path.exists(os.path.join(srcdir, 'interpreter')):
            log.info('No Cling source found ... downloading with "python create_src_directory.py"')
            if subprocess.call([sys.executable, 'create_src_directory.py']) != 0:
                log.error('ERROR: the Cling source directory "%s" does not exist' % os.path.join(srcdir, 'interpreter'))
                log.error('Please run "python create_src_directory.py" first.')
                sys.exit(1)

        if not os.path.exists(builddir):
            log.info('Creating build directory %s ...' % builddir)
            os.makedirs(builddir)

        # get C++ standard to use, if set
        try:
            stdcxx = os.environ['STDCXX']
        except KeyError:
            if is_manylinux():
                stdcxx = '17'
            else:
                stdcxx = '20'

        if not stdcxx in ['14', '17', '20']:
            log.fatal('FATAL: envar STDCXX should be one of 14, 17, or 20')
            sys.exit(1)

        stdcxx='-DCMAKE_CXX_STANDARD='+stdcxx

        # extra optimization flags for Cling
        if not 'EXTRA_CLING_ARGS' in os.environ:
            has_avx = False
            if not is_manylinux():
                try:
                    for line in open('/proc/cpuinfo', 'r'):
                        if 'avx' in line:
                            has_avx = True
                            break
                except Exception:
                    try:
                        cli_arg = subprocess.check_output(['sysctl', 'machdep.cpu.features'])
                        has_avx = 'avx' in cli_arg.decode("utf-8").strip().lower()
                    except Exception:
                        pass
            extra_args = '-O2'
            if has_avx: extra_args += ' -mavx'
            os.putenv('EXTRA_CLING_ARGS', extra_args)

        CMAKE_COMMAND = ['cmake', srcdir, '-Wno-dev',
                stdcxx, '-DLLVM_ENABLE_TERMINFO=0', '-DLLVM_ENABLE_ASSERTIONS=0',
                '-Dminimal=ON', '-Dbuiltin_cling=ON', '-Druntime_cxxmodules=OFF', '-Dbuiltin_zlib=ON']
        if 'darwin' in sys.platform:
            CMAKE_COMMAND.append('-Dlibcxx=ON')
        CMAKE_COMMAND.append('-DCMAKE_BUILD_TYPE='+get_build_type())
        if 'win32' in sys.platform:
            import platform
            bits = platform.architecture()[0]
            if '64' in bits:
                CMAKE_COMMAND += ['-A x64', '-Thost=x64', '-DCMAKE_GENERATOR_PLATFORM=x64']
            elif '32' in bits:
                CMAKE_COMMAND += ['-Thost=x86', '-DCMAKE_GENERATOR_PLATFORM=win32']
        elif 'darwin' in sys.platform:
            import platform
            if 'arm64' in platform.machine():
                CMAKE_COMMAND += ['-DLLVM_TARGETS_TO_BUILD=ARM;AArch64;NVPTX']
        CMAKE_COMMAND.append('-DCMAKE_INSTALL_PREFIX='+prefix)

        log.info('Running cmake for cppyy-cling: %s', ' '.join(CMAKE_COMMAND))
        if subprocess.call(CMAKE_COMMAND, cwd=builddir) != 0:
            raise DistutilsSetupError('Failed to configure cppyy-cling')

        # use $MAKE to build if it is defined
        env_make = os.getenv('MAKE')
        if not env_make:
            build_cmd = 'cmake'
            # default to using all available cores (x2 if hyperthreading enabled)
            nprocs = os.getenv("MAKE_NPROCS") or '0'
            try:
                nprocs = int(nprocs)
            except ValueError:
                log.warn("Integer expected for MAKE_NPROCS, but got %s (ignored)", nprocs)
                nprocs = 0
            if nprocs < 1:
                nprocs = multiprocessing.cpu_count()
            build_args = ['--build', '.', '--config', get_build_type(), '--']
            if 'win32' in sys.platform:
                build_args.append('/maxcpucount:' + str(nprocs))
            else:
                build_args.append('-j' + str(nprocs))
        else:
            build_args = env_make.split()
            build_cmd, build_args = build_args[0], build_args[1:]
        log.info('Now building cppyy-cling and dependencies ...')
        if env_make: os.unsetenv("MAKE")
        if subprocess.call([build_cmd] + build_args, cwd=builddir) != 0:
            raise DistutilsSetupError('Failed to build cppyy-cling')
        if env_make: os.putenv('MAKE', env_make)

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
        log.info('Now installing cppyy-cling into cppyy_backend')
        builddir = get_builddir()
        if not os.path.exists(builddir):
            raise DistutilsSetupError('Failed to find build dir!')

        # use $MAKE to install if it is defined
        env_make = os.getenv("MAKE")
        if not env_make:
            install_cmd = 'cmake'
            install_args = ['--build', '.', '--config', get_build_type(), '--target', 'install']
        else:
            install_args = env_make.split()
            install_cmd, install_args = install_args[0], install_args[1:]+['install']

        prefix = get_prefix()
        log.info('Now creating installation under %s ...', prefix)
        if env_make: os.unsetenv("MAKE")
        if subprocess.call([install_cmd] + install_args, cwd=builddir) != 0:
            raise DistutilsSetupError('Failed to install cppyy-cling')
        if env_make: os.putenv("MAKE", env_make)

     # remove allDict.cxx.pch as it's not portable (rebuild on first run, see cppyy)
        log.info('removing allDict.cxx.pch')
        os.remove(os.path.join(get_prefix(), 'etc', 'allDict.cxx.pch'))
     # fix-up the standard reported by cling-config (which calls root-config): if C++14
     # was requested, LLVM16 (and thus ROOT) will up it 17; and for manylinux,# which
     # was build with 17, reset the default cxxversion to 20 if no user override
        update_cxxversion = os.environ.get('STDCXX', None)
        if update_cxxversion is None and is_manylinux():
            update_cxxversion = '2a'

        if update_cxxversion is not None:
          # for manylinux, reset the default cxxversion to 20 if no user override
            log.info('updating cling-config to C++%s' % update_cxxversion)
            inp = os.path.join(get_prefix(), 'bin', 'root-config')
            outp = inp+'.new'
            outfile = open(outp, 'w')
            for line in open(inp).readlines():
                if line.find('cxxversionflag=', 0, 15) == 0:
                    line = 'cxxversionflag="-std=c++%s "\n' % update_cxxversion
                elif line.find('features=', 0, 9) == 0:
                    pcxx = line.find('cxx')
                    if 0 < pcxx:
                        line = line[:pcxx+3] + update_cxxversion + line[pcxx+5:]
                outfile.write(line)
            outfile.close()
            os.replace(outp, inp)
            os.chmod(inp, stat.S_IMODE(os.lstat(inp).st_mode) | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

            log.info('updating allCppflags.txt to C++%s' % update_cxxversion)
            inp = os.path.join(get_prefix(), 'etc', 'dictpch', 'allCppflags.txt')
            outp = inp+'.new'
            outfile = open(outp, 'w')
            for line in open(inp).readlines():
                if '-std=' == line[:5]:
                    line = '-std=c++%s\n' % update_cxxversion
                outfile.write(line)
            outfile.close()
            os.replace(outp, inp)

            log.info('updating compiledata.h to C++{update_cxxversion}')
            inp = os.path.join(get_prefix(), 'include', 'compiledata.h')
            outp = inp+'.new'
            outfile = open(outp, 'w')
            for line in open(inp).readlines():
                pstd = line.find('-std=c++')
                if 0 < pstd:
                     line = line[pstd+8] + update_cxxversion + line[pstd+10:]
                outfile.write(line)
            outfile.close()
            os.replace(outp, inp)

        install_path = self._get_install_path()
        log.info('Copying installation to: %s ...', install_path)
        self.copy_tree(os.path.join(get_prefix(), os.path.pardir), install_path)

        log.info('Install finished')

    def get_outputs(self):
        outputs = _install.get_outputs(self)
        outputs.append(os.path.join(self._get_install_path(), 'cppyy_backend'))
        version = find_version('python', 'cppyy_backend', '_version.py')
        outputs.append(os.path.join(
            self._get_install_path(), 'cppyy_backend', 'etc', 'allDict.cxx.pch.'+str(version)))
        return outputs


cmdclass = {
        'build': my_cmake_build,
        'clean': my_clean,
        'install': my_install }

if has_wheel:
    class my_bdist_wheel(_bdist_wheel):
        def finalize_options(self):
         # this is a universal, but platform-specific package; a combination
         # that wheel does not recognize, thus simply fool it
            from distutils.util import get_platform
            self.plat_name = get_platform()
            _bdist_wheel.finalize_options(self)
            self.root_is_pure = True

        def write_record(self, bdist_dir, distinfo_dir):
            _bdist_wheel.write_record(self, bdist_dir, distinfo_dir)

         # add allDict.cxx.pch to record
            record_path = os.path.join(distinfo_dir, 'RECORD')
            with open(record_path, 'a') as record_file:
                record_file.write(os.path.join('cppyy_backend', 'etc', 'allDict.cxx.pch')+',,\n')

    cmdclass['bdist_wheel'] = my_bdist_wheel


#
# customized distribition to disable binaries
#
class MyDistribution(Distribution):
    def run_commands(self):
        # disable bdist_egg as it only packages the python code, skipping the build
        if not is_manylinux():
            for cmd in self.commands:
                if cmd != 'bdist_egg':
                    self.run_command(cmd)
                else:
                    log.info('Command "%s" is disabled', cmd)
                    cmd_obj = self.get_command_obj(cmd)
                    cmd_obj.get_outputs = lambda: None
        else:
            return Distribution.run_commands(self)


setup(
    name='cppyy-cling',
    description='Re-packaged Cling, as backend for cppyy',
    long_description=long_description,
    url='https://root.cern.ch/cling',

    maintainer='Wim Lavrijsen',
    maintainer_email='WLavrijsen@lbl.gov',

    version=find_version('python', 'cppyy_backend', '_version.py'),

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

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: C',
        'Programming Language :: C++',

        'Natural Language :: English'
    ],

    keywords='interpreter development',

    setup_requires=['wheel'],

    package_data={'cppyy_backend': ['cmake/*.cmake', 'pkg_templates/*.in', 'pkg_templates/*.py']},

    package_dir={'': 'python'},
    packages=find_packages('python', include=['cppyy_backend']),

    cmdclass=cmdclass,
    distclass=MyDistribution,

    entry_points={
        "console_scripts": [
            "cling-config = cppyy_backend._cling_config:main",
            "genreflex = cppyy_backend._genreflex:main",
            "rootcling = cppyy_backend._rootcling:main",
            "cppyy-generator = cppyy_backend._cppyy_generator:main",
        ],
    },

    zip_safe=False,
)
