set(ROOT_ARCHITECTURE macosx)
set(ROOT_PLATFORM macosx)

set(SYSLIBS "-lm ${EXTRA_LDFLAGS} ${FINK_LDFLAGS} ${CMAKE_THREAD_LIBS_INIT} -ldl")
set(XLIBS "${XPMLIBDIR} ${XPMLIB} ${X11LIBDIR} -lXext -lX11")
set(CILIBS "-lm ${EXTRA_LDFLAGS} ${FINK_LDFLAGS} -ldl")
#set(CRYPTLIBS "-lcrypt")
set(CMAKE_M_LIBS -lm)

#---This is needed to help CMake to locate the X11 headers in the correct place and not under /usr/include
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} /usr/X11R6)
#---------------------------------------------------------------------------------------------------------

if (CMAKE_SYSTEM_NAME MATCHES Darwin)
  EXECUTE_PROCESS(COMMAND sw_vers "-productVersion"
                  COMMAND cut -d . -f 1-2
                  OUTPUT_VARIABLE MACOSX_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
  MESSAGE(STATUS "Found a Mac OS X System ${MACOSX_VERSION}")
  EXECUTE_PROCESS(COMMAND sw_vers "-productVersion"
                  COMMAND cut -d . -f 2
                  OUTPUT_VARIABLE MACOSX_MINOR OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(MACOSX_VERSION VERSION_GREATER 10.7 AND ${CMAKE_CXX_COMPILER_ID} MATCHES Clang)
    set(libcxx ON CACHE BOOL "Build using libc++" FORCE)
  endif()

  if(${MACOSX_MINOR} GREATER 4)
    #TODO: check haveconfig and rpath -> set rpath true
    #TODO: check Thread, define link command
    #TODO: more stuff check configure script
    execute_process(COMMAND /usr/sbin/sysctl machdep.cpu.extfeatures OUTPUT_VARIABLE SYSCTL_OUTPUT)
    if(${SYSCTL_OUTPUT} MATCHES 64)
       MESSAGE(STATUS "Found a 64bit system")
       set(ROOT_ARCHITECTURE macosx64)
       SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
       SET(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -m64")
       SET(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -m64")
       SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
       SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64")
       SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -m64")
    else(${SYSCTL_OUTPUT} MATCHES 64)
       MESSAGE(STATUS "Found a 32bit system")
       SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
       SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
       SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -m32")
    endif(${SYSCTL_OUTPUT} MATCHES 64)
  endif()

  if(MACOSX_VERSION VERSION_GREATER 10.6)
    set(MACOSX_SSL_DEPRECATED ON)
  endif()
  if(MACOSX_VERSION VERSION_GREATER 10.7)
    set(MACOSX_ODBC_DEPRECATED ON)
  endif()
  if(MACOSX_VERSION VERSION_GREATER 10.8)
    set(MACOSX_GLU_DEPRECATED ON)
    set(MACOSX_KRB5_DEPRECATED ON)
  endif()
  if(MACOSX_VERSION VERSION_GREATER 10.9)
    set(MACOSX_LDAP_DEPRECATED ON)
  endif()

  if (CMAKE_COMPILER_IS_GNUCXX)
     message(STATUS "Found GNU compiler collection")

     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -W -Wshadow -Wall -Woverloaded-virtual -fsigned-char -fno-common")
     SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -W -Wall -fsigned-char -fno-common")
     SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")
     SET(CINT_CXX_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__ROOT -DG__REDIRECTIO -DG__OSFDLL -DG__STD_EXCEPTION")
     SET(CINT_C_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__ROOT -DG__REDIRECTIO -DG__OSFDLL -DG__STD_EXCEPTION")

     SET(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -single_module -Wl,-dead_strip_dylibs")
     SET(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -single_module -Wl,-dead_strip_dylibs")

     set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} -bind_at_load -m64")
     set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -bind_at_load -m64")

     # Select flags.
     set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
     set(CMAKE_CXX_FLAGS_RELEASE        "-O2 -DNDEBUG")
     set(CMAKE_CXX_FLAGS_OPTIMIZED      "-O3 -ffast-math -DNDEBUG")
     set(CMAKE_CXX_FLAGS_DEBUG          "-g")
     set(CMAKE_CXX_FLAGS_DEBUGFULL      "-g3")
     set(CMAKE_CXX_FLAGS_PROFILE        "-g3 -ftest-coverage -fprofile-arcs")
     set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -DNDEBUG")
     set(CMAKE_C_FLAGS_RELEASE          "-O2 -DNDEBUG")
     set(CMAKE_C_FLAGS_OPTIMIZED        "-O3 -ffast-math -DNDEBUG")
     set(CMAKE_C_FLAGS_DEBUG            "-g")
     set(CMAKE_C_FLAGS_DEBUGFULL        "-g3")
     set(CMAKE_C_FLAGS_PROFILE          "-g3 -ftest-coverage -fprofile-arcs")

     #settings for cint
     set(CPPPREP "${CXX} -E -C")
     set(CXXOUT "-o ")
     set(EXEEXT "")
     set(SOEXT "so")

  elseif(${CMAKE_CXX_COMPILER_ID} MATCHES Clang)

     message(STATUS "Found LLVM compiler collection")

     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -W -Wall -Woverloaded-virtual -fsigned-char -fno-common -Qunused-arguments")
     SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -W -Wall -fsigned-char -fno-common -Qunused-arguments")
     if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8)
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow")
     endif()

     SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")
     SET(CINT_CXX_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__ROOT -DG__REDIRECTIO -DG__OSFDLL -DG__STD_EXCEPTION")
     SET(CINT_C_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__ROOT -DG__REDIRECTIO -DG__OSFDLL -DG__STD_EXCEPTION")

     SET(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -single_module -Wl,-dead_strip_dylibs")
     SET(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -single_module -Wl,-dead_strip_dylibs")

     set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} -bind_at_load -m64")
     set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -bind_at_load -m64")

     # Select flags.
     set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
     set(CMAKE_CXX_FLAGS_RELEASE        "-O2 -DNDEBUG")
     set(CMAKE_CXX_FLAGS_OPTIMIZED      "-O3 -ffast-math -DNDEBUG")
     set(CMAKE_CXX_FLAGS_DEBUG          "-g")
     set(CMAKE_CXX_FLAGS_DEBUGFULL      "-g3")
     set(CMAKE_CXX_FLAGS_PROFILE        "-g3 -ftest-coverage -fprofile-arcs")
     set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -DNDEBUG")
     set(CMAKE_C_FLAGS_RELEASE          "-O2 -DNDEBUG")
     set(CMAKE_C_FLAGS_OPTIMIZED        "-O3 -ffast-math -DNDEBUG")
     set(CMAKE_C_FLAGS_DEBUG            "-g")
     set(CMAKE_C_FLAGS_DEBUGFULL        "-g3")
     set(CMAKE_C_FLAGS_PROFILE          "-g3 -ftest-coverage -fprofile-arcs")

     #settings for cint
     set(CPPPREP "${CXX} -E -C")
     set(CXXOUT "-o ")
     set(EXEEXT "")
     set(SOEXT "so")
  else()
    MESSAGE(FATAL_ERROR "There is no setup for this compiler with ID=${CMAKE_CXX_COMPILER_ID} up to now. Don't know what to do. Stop cmake at this point.")
  endif()

  #---Set Linker flags----------------------------------------------------------------------
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}  -mmacosx-version-min=${MACOSX_VERSION} -Wl,-rpath,@loader_path/../lib")


else (CMAKE_SYSTEM_NAME MATCHES Darwin)
  MESSAGE(FATAL_ERROR "There is no setup for this this Apple system up to now. Don't know waht to do. Stop cmake at this point.")
endif (CMAKE_SYSTEM_NAME MATCHES Darwin)

#---Avoid puting the libraires and executables in different configuration locations
if(CMAKE_GENERATOR MATCHES Xcode)
  foreach( _conf ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${_conf} _conf )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_conf} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} )
    set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_conf} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
    set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_conf} ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} )
  endforeach()
endif()


