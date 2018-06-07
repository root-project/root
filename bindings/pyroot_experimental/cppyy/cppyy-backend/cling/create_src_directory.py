#!/usr/bin/env python
from __future__ import print_function

import os, sys, subprocess
import shutil, tarfile
try:
    import urllib2
except ModuleNotFoundError:
    import urllib.request as urllib2  # p3


DEBUG_TESTBUILD = False

TARBALL_CACHE_DIR = 'releases'

ROOT_KEEP = ['build', 'cmake', 'config', 'core', 'etc', 'interpreter',
             'io', 'LICENSE', 'Makefile', 'CMakeLists.txt', 'math',
             'main'] # main only needed in more recent root b/c of rootcling
ROOT_CORE_KEEP = ['CMakeLists.txt', 'base', 'clib', 'clingutils', 'cont',
                  'dictgen', 'foundation', 'lz4', 'lzma', 'macosx', 'meta',
                  'metacling', 'metautils', 'rootcling_stage1', 'textinput',
                  'thread', 'unix', 'utils', 'winnt', 'zip', 'pcre']
ROOT_IO_KEEP = ['CMakeLists.txt', 'io', 'rootpcm']
ROOT_MATH_KEEP = ['CMakeLists.txt', 'mathcore']
ROOT_ETC_KEEP = ['Makefile.arch', 'class.rules', 'cmake', 'dictpch',
                 'gdb-backtrace.sh', 'gitinfo.txt', 'helgrind-root.supp',
                 'hostcert.conf', 'plugins', 'system.plugins-ios',
                 'valgrind-root-python.supp', 'valgrind-root.supp', 'vmc']
ROOT_PLUGINS_KEEP = ['TVirtualStreamerInfo']

ROOT_EXPLICIT_REMOVE = ['core/base/v7', 'math/mathcore/v7', 'io/io/v7']


ERR_RELEASE_NOT_FOUND = 2


def get_root_version(try_recover=True):
    import pkg_resources
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'python')
    dists = pkg_resources.find_distributions(path)
    try:
        cppyy_cling = [d for d in dists if d.key == 'cppyy-cling'][0]
        version = cppyy_cling.version
    except IndexError:
        if try_recover and os.path.exists('setup.py'):
            print('No egg_info ... running "python setup.py egg_info"')
            if subprocess.call(['python', 'setup.py', 'egg_info']) != 0:
                print('ERROR: creation of egg_info failed ... giving up')
                sys.exit(2)
            return get_root_version(False)
        else:
            print('ERROR: cannot determine version. Please run "python setup.py egg_info" first.')
            sys.exit(1)
    #
    parts = version.split('.', 3)
    major, minor, patch = map(int, parts[:3])
    root_version = '%d.%02d.%02d' % (major, minor, patch)
    return root_version


ROOT_VERSION = get_root_version()

#
## ROOT source pull and cleansing
#
def clean_directory(directory, keeplist, trim_cmake=True):
    removed_entries = []
    for entry in os.listdir(directory):
        if entry[0] == '.' or entry in keeplist:
            continue
        removed_entries.append(entry)
        entry = os.path.join(directory, entry)
        print('now removing', entry)
        if os.path.isdir(entry):
            shutil.rmtree(entry)
        else:
            os.remove(entry)

    if not trim_cmake:
        return

    # now take the removed entries out of the CMakeLists.txt
    if removed_entries:
        inp = os.path.join(directory, 'CMakeLists.txt')
        print('trimming', inp)
        outp = inp+'.new'
        new_cml = open(outp, 'w')
        for line in open(inp).readlines():
            if ('add_subdirectory' in line) or\
               ('COMMAND' in line and 'copy' in line) or\
               ('ROOT_ADD_TEST_SUBDIRECTORY' in line) or\
               ('install(DIRECTORY' in line):
                for sub in removed_entries:
                    if sub in line:
                        line = '#'+line
                        break
            new_cml.write(line)
        new_cml.close()
        os.rename(outp, inp)
    else:
        print('reusing existing %s/CMakeLists.txt' % (directory,))
 

if not os.path.exists(TARBALL_CACHE_DIR):
    os.mkdir(TARBALL_CACHE_DIR)

fn = 'root_v%s.source.tar.gz' % ROOT_VERSION
addr = 'https://root.cern.ch/download/'+fn
if not os.path.exists(os.path.join(TARBALL_CACHE_DIR, fn)):
    try:
        print('retrieving', fn)
        if sys.hexversion < 0x3000000:
            output_fn = fn
        else:
            output_fn = bytes(fn, 'utf-8')
        resp = urllib2.urlopen(addr, output_fn)
        out = open(os.path.join(TARBALL_CACHE_DIR, fn), 'wb')
        out.write(resp.read())
        out.close()
    except urllib2.HTTPError:
        print('release %s not found' % ROOT_VERSION)
        sys.exit(ERR_RELEASE_NOT_FOUND)
else:
    print('reusing', fn, 'from local directory')


fn = os.path.join(TARBALL_CACHE_DIR, fn)
pkgdir = os.path.join('root-' + ROOT_VERSION)
if not os.path.exists(pkgdir):
    print('now extracting', ROOT_VERSION)
    tf = tarfile.TarFile.gzopen(fn)
    tf.extractall()
    tf.close()
else:
    print('reusing existing directory', pkgdir)

# remove everything except for the listed set of libraries
try:
    shutil.rmtree('src')
except OSError:
    pass
os.chdir(pkgdir)
clean_directory(os.path.curdir, ROOT_KEEP)
clean_directory('core',         ROOT_CORE_KEEP)
clean_directory('etc',          ROOT_ETC_KEEP, trim_cmake=False)
clean_directory('etc/plugins',  ROOT_PLUGINS_KEEP, trim_cmake=False)
clean_directory('io',           ROOT_IO_KEEP)
clean_directory('math',         ROOT_MATH_KEEP)


# trim main (only need rootcling)
print('trimming main')
for entry in os.listdir('main/src'):
    if entry != 'rootcling.cxx':
        os.remove('main/src/'+entry)
inp = 'main/CMakeLists.txt'
outp = inp+'.new'
new_cml = open(outp, 'w')
for line in open(inp).readlines():
    if ('ROOT_EXECUTABLE' in line or\
        'SET_TARGET_PROPERTIES' in line) and\
       not 'rootcling' in line:
        line = '#'+line
    new_cml.write(line)
new_cml.close()
os.rename(outp, inp)

# trim core (gGLManager crashes on call)
print('trimming main')
os.remove("core/base/src/TVirtualGL.cxx")
os.remove("core/base/inc/TVirtualGL.h")

# remove afterimage and ftgl explicitly
print('trimming externals')
for cmf in ['AfterImage', 'FTGL']:
    fname = 'cmake/modules/Find%s.cmake' % (cmf,)
    if os.path.exists(fname):
        os.remove(fname)
inp = 'cmake/modules/SearchInstalledSoftware.cmake'
outp = inp+'.new'
now_stripping = False
new_cml = open(outp, 'w')
for line in open(inp).readlines():
    if '#---Check for ftgl if needed' == line[0:28] or\
       '#---Check for AfterImage' == line[0:24]:
        now_stripping = True
    elif '#---Check' == line[0:9]:
        now_stripping = False
    if now_stripping:
        line = '#'+line
    new_cml.write(line)
new_cml.close()
os.rename(outp, inp)

inp = 'cmake/modules/RootBuildOptions.cmake'
outp = inp+'.new'
new_cml = open(outp, 'w')
for line in open(inp).readlines():
    if 'ROOT_BUILD_OPTION(builtin_ftgl' in line or\
       'ROOT_BUILD_OPTION(builtin_afterimage' in line:
        line = '#'+line
    new_cml.write(line)
new_cml.close()
os.rename(outp, inp)

# strip freetype
inp = 'cmake/modules/SearchInstalledSoftware.cmake'
outp = inp+'.new'
new_cml = open(outp, 'w')
for line in open(inp).readlines():
    if '#---Check for Freetype' == line[0:22]:
        now_stripping = True
    elif '#---Check for PCRE' == line[0:18]:
        now_stripping = False
    if now_stripping or 'builtin_freetype' in line:
        line = '#'+line
    new_cml.write(line)
new_cml.close()
os.rename(outp, inp)

# remove testing and examples
print('trimming testing')
inp = 'CMakeLists.txt'
outp = inp+'.new'
now_stripping = False
new_cml = open(outp, 'w')
for line in open(inp).readlines():
    if '#---Configure Testing using CTest' == line[0:33] or\
       '#---hsimple.root' == line[0:16]:
        now_stripping = True
    elif '#---Packaging' == line[0:13] or\
         '#---version' == line[0:11]:
        now_stripping = False
    if now_stripping:
        line = '#'+line
    new_cml.write(line)
new_cml.close()
os.rename(outp, inp)

print('trimming RootCPack')
inp = 'cmake/modules/RootCPack.cmake'
outp = inp+'.new'
new_cml = open(outp, 'w')
for line in open(inp):
    if 'README.txt' in line:
        line = '#'+line
    new_cml.write(line)
new_cml.close()
os.rename(outp, inp)


# some more explicit removes:
for dir_to_remove in ROOT_EXPLICIT_REMOVE:
    try:
        shutil.rmtree(dir_to_remove)
    except OSError:
        pass


# special fixes
inp = 'core/base/src/TVirtualPad.cxx'
outp = inp+'.new'
new_cml = open(outp, 'w')
for line in open(inp):
    if '#include "X3DBuffer.h"' == line[0:22]:
        line = """//#include "X3DBuffer.h"
typedef struct _x3d_sizeof_ {
   int  numPoints;
   int  numSegs;
   int  numPolys;
} Size3D;
"""
    new_cml.write(line)
new_cml.close()
os.rename(outp, inp)

inp = 'core/unix/src/TUnixSystem.cxx'
outp = inp+'.new'
new_cml = open(outp, 'w')
for line in open(inp):
    if '#include "TSocket.h"' == line[0:20]:
        line = """//#include "TSocket.h"
enum ESockOptions {
   kSendBuffer,        // size of send buffer
   kRecvBuffer,        // size of receive buffer
   kOobInline,         // OOB message inline
   kKeepAlive,         // keep socket alive
   kReuseAddr,         // allow reuse of local portion of address 5-tuple
   kNoDelay,           // send without delay
   kNoBlock,           // non-blocking I/O
   kProcessGroup,      // socket process group (used for SIGURG and SIGIO)
   kAtMark,            // are we at out-of-band mark (read only)
   kBytesToRead        // get number of bytes to read, FIONREAD (read only)
};

enum ESendRecvOptions {
   kDefault,           // default option (= 0)
   kOob,               // send or receive out-of-band data
   kPeek,              // peek at incoming message (receive only)
   kDontBlock          // send/recv as much data as possible without blocking
};
"""
    new_cml.write(line)
new_cml.close()
os.rename(outp, inp)

inp = 'math/mathcore/src/Fitter.cxx'
if os.path.exists(inp):
    outp = inp+'.new'
    new_cml = open(outp, 'w')
    for line in open(inp):
        if '#include "TF1.h"' in line:
            continue
        new_cml.write(line)
    new_cml.close()
    os.rename(outp, inp)

# done
os.chdir(os.path.pardir)

# debugging: run a test build
if DEBUG_TESTBUILD:
    print('running a debug test build')
    tb = "test_builddir"
    if os.path.exists(tb):
        shutil.rmtree(tb)
    os.mkdir(tb)
    os.chdir(tb)
    os.system('cmake ../%s -DCMAKE_INSTALL_PREFIX=../install -Dminimal=ON -Dasimage=OFF' % pkgdir)
    os.system('make -j 32')


#
## package creation
#
countdown = 0

print('adding src ... ')
if not os.path.exists('src'):
    os.mkdir('src')
for entry in os.listdir(pkgdir):
    fullp = os.path.join(pkgdir, entry)
    if entry[0] == '.':
        continue
    dest = os.path.join('src', entry)
    if os.path.isdir(fullp):
        if not os.path.exists(dest):
            shutil.copytree(fullp, dest)
    else:
        if not os.path.exists(dest):
            shutil.copy2(fullp, dest)

#
## apply patches
#
os.system('patch -p1 < patches/metacling.diff')
os.system('patch -p1 < patches/scanner.diff')
os.system('patch -p1 < patches/scanner_2.diff')
os.system('patch -p1 < patches/faux_typedef.diff')
os.system('patch -p1 < patches/template_fwd.diff')
os.system('patch -p1 < patches/dep_template.diff')
os.system('patch -p1 < patches/no_long64_t.diff')
os.system('patch -p1 < patches/using_decls.diff')
os.system('patch -p1 < patches/sfinae.diff')

# done!
