from distutils.core import setup
from distutils import sysconfig
from distutils import fancy_getopt 
import os, sys, getopt
import distutils


top_dir = sysconfig.PREFIX
# this stupid hack is necessary because it is not possible to get
# the --prefix option from the command line through distutils
for i in range(len(sys.argv[1:])) :
  opt = sys.argv[1:][i]
  lopt = opt.split('=')
  if lopt[0] == '--prefix':
    if len(lopt) > 1: top_dir = os.path.realpath(lopt[1])
    else : top_dir = os.path.realpath(sys.argv[1:][i+1])

setup( name = 'genreflex', version = '0.1', py_modules =
       ['genreflex.gccdemangler', 'genreflex.gencapa',
       'genreflex.gendict', 'genreflex.genheader',
       'genreflex.genreflex', 'genreflex.selclass']

       )

genreflex_pydir = sysconfig.get_python_lib(prefix=top_dir) + os.sep + 'genreflex' + os.sep

s = ''
genreflex_file_name = 'genreflex'
if (sys.platform == 'win32'):
  file_name += '.bat'
  s += '@echo off\n'
  s += 'python %sgenreflex.py %*\n' % genreflex_pydir
else:
  s += '#!/bin/csh -f\n'
  s += 'python %sgenreflex.py $*\n' % genreflex_pydir

genreflex_shell_name = top_dir+os.sep+'bin'+os.sep+genreflex_file_name
if os.path.isfile(genreflex_shell_name) : os.remove(genreflex_shell_name)

f = open(genreflex_shell_name,'w')
f.write(s)
f.close()

if (sys.platform != 'win32') : os.chmod(genreflex_shell_name,0755)
