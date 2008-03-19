from os import curdir
from os.path import realpath

opts = Options('custom.py')

opts.Add(BoolOption   ('test',    'Set to 1 if tests shall be compiled', 0))
opts.Add(PathOption   ('prefix',  'The installation directory of Reflex', realpath(curdir)))
opts.Add(PackageOption('gccxml',  'The >binary< directory of gccxml',     'no'))
opts.Add(PackageOption('cppunit', 'The >root< directory of CppUnit',      'no'))

env = Environment(options=opts)
Export('env')

Help(opts.GenerateHelpText(env))

SConscript(['src/SConscript',
            'inc/SConscript',
            'python/SConscript'])

if env['test'] : SConscript('test/SConscript')

env.Alias('install', env['prefix'])
