#---Post actions to use builtin GSL----------------------------------------------------
if(builtin_gsl)
  if(TARGET MathMore)
    add_dependencies(MathMore GSL)
  endif()
  ExternalProject_Get_Property(GSL install_dir)
  install(DIRECTORY ${install_dir}/lib/ DESTINATION lib FILES_MATCHING PATTERN "libgsl*")
  #install(FILES ${install_dir}/lib/libgslcblas.so  
  #              ${install_dir}/lib/libgsl.so  
  #              DESTINATION lib)
endif()

#---Post actions to use builtin CFITSIO------------------------------------------------
if(builtin_cfitsio)
  if(TARGET FITSIO)
    add_dependencies(FITSIO CFITSIO)
  endif()
  #install(FILES ${install_dir}/lib/libcfitsio.so  
  #              DESTINATION lib)
endif()
