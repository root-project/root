set(ROOT_ARCHITECTURE linux)
set(ROOT_PLATFORM linux)

execute_process(COMMAND uname -m OUTPUT_VARIABLE SYSCTL_OUTPUT)

if(${SYSCTL_OUTPUT} MATCHES x86_64)
  message(STATUS "Found a 64bit system")
  set(BIT_ENVIRONMENT "-m64")
  set(SPECIAL_CINT_FLAGS "-DG__64BIT")
  if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "Found GNU compiler collection")
    set(ROOT_ARCHITECTURE linuxx8664gcc)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
    message(STATUS "Found CLANG compiler")
    set(ROOT_ARCHITECTURE linuxx8664gcc)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
    set(ROOT_ARCHITECTURE linuxx8664icc)
  else()
    message(FATAL_ERROR "There is no Setup for this compiler up to now. Don't know what to do. Stop cmake at this point.")
  endif()
else()
  message(STATUS "Found a 32bit system")
  set(BIT_ENVIRONMENT "-m32")
  set(SPECIAL_CINT_FLAGS "")
  if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "Found GNU compiler collection")
    set(ROOT_ARCHITECTURE linux)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
    message(STATUS "Found CLANG compiler")
    set(ROOT_ARCHITECTURE linux)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL Intel)
    set(ROOT_ARCHITECTURE linuxicc)
  else()
    message(FATAL_ERROR "There is no Setup for this compiler up to now. Don't know what to do. Stop cmake at this point.")
  endif()
endif()

set(SYSLIBS "-lm -ldl ${CMAKE_THREAD_LIBS_INIT} -rdynamic")
set(XLIBS "${XPMLIBDIR} ${XPMLIB} ${X11LIBDIR} -lXext -lX11")
set(CILIBS "-lm -ldl -rdynamic")
set(CRYPTLIBS "-lcrypt")
set(CMAKE_M_LIBS -lm)

if(CMAKE_COMPILER_IS_GNUCXX)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe ${BIT_ENVIRONMENT} -Wall -W -Woverloaded-virtual -fPIC")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe ${BIT_ENVIRONMENT} -Wall -W -fPIC")

  set(CMAKE_Fortran_FLAGS "${CMAKE_FORTRAN_FLAGS} ${BIT_ENVIRONMENT} -std=legacy")

  set(CINT_CXX_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__OSFDLL -DG__ROOT -DG__REDIRECTIO -DG__STD_EXCEPTION ${SPECIAL_CINT_FLAGS}")
  set(CINT_C_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__OSFDLL -DG__ROOT -DG__REDIRECTIO -DG__STD_EXCEPTION ${SPECIAL_CINT_FLAGS}")

  set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS}")
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS}")

  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

  # Select flags.
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE        "-O2 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG          "-g  -fno-reorder-blocks -fno-schedule-insns -fno-inline")
  set(CMAKE_CXX_FLAGS_DEBUGFULL      "-g3 -fno-inline")
  set(CMAKE_CXX_FLAGS_PROFILE        "-g3 -fno-inline -ftest-coverage -fprofile-arcs")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -DNDEBUG")
  set(CMAKE_C_FLAGS_RELEASE          "-O2 -DNDEBUG")
  set(CMAKE_C_FLAGS_DEBUG            "-g  -fno-reorder-blocks -fno-schedule-insns -fno-inline")
  set(CMAKE_C_FLAGS_DEBUGFULL        "-g3 -fno-inline")
  set(CMAKE_C_FLAGS_PROFILE          "-g3 -fno-inline -ftest-coverage -fprofile-arcs")

  #Settings for cint
  if (NOT (GCC_MAJOR LESS 6))
    set(CPPPREP "${CXX} -std=c++98 -E -C")
  else()
    set(CPPPREP "${CXX} -E -C")
  endif()

  set(CXXOUT "-o ")
  set(EXPLICITLINK "no") #TODO

  set(EXEEXT "")
  set(SOEXT "so")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe ${BIT_ENVIRONMENT} -Wall -W -Woverloaded-virtual -fPIC")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe ${BIT_ENVIRONMENT} -Wall -W -fPIC")

  set(CMAKE_Fortran_FLAGS "${CMAKE_FORTRAN_FLAGS} ${BIT_ENVIRONMENT} -std=legacy")

  set(CINT_CXX_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__OSFDLL -DG__ROOT -DG__REDIRECTIO -DG__STD_EXCEPTION ${SPECIAL_CINT_FLAGS}")
  set(CINT_C_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__OSFDLL -DG__ROOT -DG__REDIRECTIO -DG__STD_EXCEPTION ${SPECIAL_CINT_FLAGS}")

  set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS}")
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS}")

  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

  # Select flags.
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE        "-O2 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG          "-g -fno-schedule-insns -fno-inline")
  set(CMAKE_CXX_FLAGS_DEBUGFULL      "-g3 -fno-inline")
  set(CMAKE_CXX_FLAGS_PROFILE        "-g3 -fno-inline -ftest-coverage -fprofile-arcs")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -DNDEBUG")
  set(CMAKE_C_FLAGS_RELEASE          "-O2 -DNDEBUG")
  set(CMAKE_C_FLAGS_DEBUG            "-g  -fno-schedule-insns -fno-inline")
  set(CMAKE_C_FLAGS_DEBUGFULL        "-g3 -fno-inline")
  set(CMAKE_C_FLAGS_PROFILE          "-g3 -fno-inline -ftest-coverage -fprofile-arcs")

  #Settings for cint
  set(CPPPREP "${CXX} -E -C")
  set(CXXOUT "-o ")
  set(EXPLICITLINK "no") #TODO

  set(EXEEXT "")
  set(SOEXT "so")

elseif(CMAKE_CXX_COMPILER_ID STREQUAL Intel)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -wd1476")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -restrict")

  set(CMAKE_Fortran_FLAGS "${CMAKE_FORTRAN_FLAGS} -fPIC")

  set(CINT_CXX_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__OSFDLL -DG__ROOT -DG__REDIRECTIO -DG__STD_EXCEPTION ${SPECIAL_CINT_FLAGS}")
  set(CINT_C_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__OSFDLL -DG__ROOT -DG__REDIRECTIO -DG__STD_EXCEPTION ${SPECIAL_CINT_FLAGS}")


  # Check icc compiler version and set compile flags according to the
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -v
                  ERROR_VARIABLE _icc_version_info ERROR_STRIP_TRAILING_WHITESPACE)

  string(REGEX REPLACE "(^V|^icc[ ]v|^icpc[ ]v)ersion[ ]([0-9]+)\\.[0-9]+.*" "\\2" ICC_MAJOR "${_icc_version_info}")
  string(REGEX REPLACE "(^V|^icc[ ]v|^icpc[ ]v)ersion[ ][0-9]+\\.([0-9]+).*" "\\2" ICC_MINOR "${_icc_version_info}")

  message(STATUS "Found ICC major version ${ICC_MAJOR}")
  message(STATUS "Found ICC minor version ${ICC_MINOR}")

  if(ICC_MAJOR EQUAL 9)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd1572")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -wd1572")
  endif()

  if(ICC_MAJOR EQUAL 10)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd1572")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -wd1572")
  endif()

  if(ICC_MAJOR EQUAL 11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BIT_ENVIRONMENT} -wd1572 -wd279")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BIT_ENVIRONMENT} -wd1572 -wd279")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${BIT_ENVIRONMENT} -Wl,--no-undefined")
  endif()

  if(ICC_MAJOR EQUAL 12)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BIT_ENVIRONMENT} -wd1572 -wd279")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BIT_ENVIRONMENT} -wd1572 -wd279")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${BIT_ENVIRONMENT} -Wl,--no-undefined")
  endif()

  if(ICC_MAJOR EQUAL 13)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BIT_ENVIRONMENT} -wd1572 -wd279")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BIT_ENVIRONMENT} -wd1572 -wd279")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${BIT_ENVIRONMENT} -Wl,--no-undefined")
  endif()

  if(ICC_MAJOR EQUAL 14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BIT_ENVIRONMENT} -wd1572 -wd279 -wd2536 -wd873 -wd1478")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BIT_ENVIRONMENT} -wd1572 -wd279 -wd2536 -wd873 -wd1478")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${BIT_ENVIRONMENT} -Wl,--no-undefined")
  endif()

  if(ICC_MAJOR EQUAL 15)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BIT_ENVIRONMENT} -wd1572 -wd279 -wd2536 -wd873 -wd1292 -wd3373 -wd597 -wd1098 -wd1478")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BIT_ENVIRONMENT} -wd1572 -wd279 -wd2536 -wd873")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${BIT_ENVIRONMENT} -Wl,--no-undefined")
  endif()

  set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS}")
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS}")


  # Select flags.
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O0 -g -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELEASE        "-O -DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG          "-g")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O0 -g -DNDEBUG")
  set(CMAKE_C_FLAGS_RELEASE          "-O -DNDEBUG")
  set(CMAKE_C_FLAGS_DEBUG            "-g")

  #Settings for cint
  set(CPPPREP "${CXX} -E -C")
  set(CXXOUT "-o ")
  set(EXPLICITLINK "no") #TODO

  set(EXEEXT "")
  set(SOEXT "so")

endif()

