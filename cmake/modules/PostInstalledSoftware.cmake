#---Post actions to use builtin GSL----------------------------------------------------
if(builtin_gsl)
  if(TARGET MathMore)
    add_dependencies(MathMore GSL)
  endif()
  ExternalProject_Get_Property(GSL install_dir)
  install(DIRECTORY ${install_dir}/lib/ DESTINATION ${CMAKE_INSTALL_LIBDIR} FILES_MATCHING PATTERN "libgsl*")
endif()

#---Post actions to use builtin CFITSIO------------------------------------------------
if(builtin_cfitsio)
  if(TARGET FITSIO)
    add_dependencies(FITSIO CFITSIO)
  endif()
endif()
