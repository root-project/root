import distutils.sysconfig, sys, os; 
#print (distutils.sysconfig.get_config_var("LDLIBRARY") or ("Python%d%d" % (sys.version_info[0],sys.version_info[1]))),
lib = os.path.join(distutils.sysconfig.get_config_var('LIBDIR'), distutils.sysconfig.get_config_var('LDLIBRARY'))
if not os.path.isfile(lib):
   lib = os.path.join(distutils.sysconfig.get_config_var('LIBPL'), distutils.sysconfig.get_config_var('LDLIBRARY'))
   if not os.path.isfile(lib):
     lib = os.path.join('/System','Library','Frameworks',distutils.sysconfig.get_config_var('LDLIBRARY'))
     if not os.path.isfile(lib):
       lib = os.path.join('/Library','Frameworks',distutils.sysconfig.get_config_var('LDLIBRARY'))
       if not os.path.isfile(lib):
         raise RuntimeError("Cannot locate Python dynamic libraries");
print lib