############################################################################
# CMakeLists.txt file for building ROOT misc/minicern package
############################################################################

if(WIN32)
  #---libminicern static library--------------------------------------------
  if(winrtdebug)
    set(F77FLAGS /MDd)
  else()
    set(F77FLAGS /MD)
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    set(FORTRAN_FLAGS /optimize:0 /nopdbfile /nologo /4Nportlib ${F77FLAGS} /recursive)
  else()
    set(FORTRAN_FLAGS /debug /nopdbfile /nologo /4Nportlib ${F77FLAGS} /recursive)
  endif()

  set(FortranSources
     ${CMAKE_CURRENT_SOURCE_DIR}/src/hbook.f 
     ${CMAKE_CURRENT_SOURCE_DIR}/src/kernlib.f
     ${CMAKE_CURRENT_SOURCE_DIR}/src/zebra.f
  )
  set(FortranObjects)
  foreach(_file ${FortranSources})
    string(REPLACE "/" "\\" _file "${_file}")
    get_filename_component(_filewe ${_file} NAME_WE)
    set(_fileobj ${_filewe}.obj)
    set(_fileobj_output ${CMAKE_CURRENT_BINARY_DIR}/minicern.dir/${CMAKE_CFG_INTDIR}/${_fileobj})
    string(REPLACE "/" "\\" _fileobject "${_fileobj_output}")
    add_custom_command(OUTPUT ${_fileobj_output}
         COMMENT "Compiling ${_file}"
         COMMAND ${CMAKE_Fortran_COMPILER} ${FORTRAN_FLAGS} /Fo${_fileobject} /c ${_file}
    )
    list(APPEND FortranObjects ${_fileobj_output})
  endforeach()

  ROOT_LINKER_LIBRARY(minicern *.c ${FortranObjects} TYPE STATIC)

else()
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fPIC")
  ROOT_LINKER_LIBRARY(minicern *.c *.f TYPE STATIC)
endif()
