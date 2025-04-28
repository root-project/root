import distutils.sysconfig as conf, sys, os; 
ldlib = conf.get_config_var('LDLIBRARY')
if ldlib:
   for libdir in [conf.get_config_var('LIBDIR'),
                  conf.get_config_var('LIBPL'),
                  os.path.join('/System','Library','Frameworks'),
                  os.path.join('/Library','Frameworks')]:
      lib = os.path.join(libdir,ldlib)
      if os.path.isfile(lib):
         break
elif 'win' in sys.platform:
   # mingw severely limited in distutils vars, construct 'by hand'
   prefix = conf.get_config_var('prefix')
   version = conf.get_config_var('VERSION')
   lib = os.path.join(prefix,'python%s.dll' % version)
if not os.path.isfile(lib):
   raise RuntimeError("Cannot locate Python dynamic libraries");
print(lib)
