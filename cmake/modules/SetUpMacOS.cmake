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
                  
  if(${MACOSX_MINOR} GREATER 4)
    #TODO: check haveconfig and rpath -> set rpath true
    #TODO: check Thread, define link command
    #TODO: more stuff check configure script
    execute_process(COMMAND sysctl machdep.cpu.extfeatures OUTPUT_VARIABLE SYSCTL_OUTPUT)
    if(${SYSCTL_OUTPUT} MATCHES 64)
       MESSAGE(STATUS "Found a 64bit system") 
       set(ROOT_ARCHITECTURE macosx64)
       SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
       SET(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -m64")
       SET(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -m64")
       SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
       SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64")
       SET(CMAKE_FORTRAN_FLAGS "${CMAKE_FORTRAN_FLAGS} -m64")
    else(${SYSCTL_OUTPUT} MATCHES 64)
       MESSAGE(STATUS "Found a 32bit system") 
       SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
       SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
       SET(CMAKE_FORTRAN_FLAGS "${CMAKE_FORTRAN_FLAGS} -m32")
    endif(${SYSCTL_OUTPUT} MATCHES 64)
  endif()
  
  if (CMAKE_COMPILER_IS_GNUCXX)
     MESSAGE(STATUS "Found GNU compiler collection")
     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -W -Wall -Woverloaded-virtual -fsigned-char -fno-common")
     SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -W -Wall -fsigned-char -fno-common")
     SET(CMAKE_Fortran_FLAGS "${CMAKE_FORTRAN_FLAGS}")
     SET(CINT_CXX_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__ROOT -DG__REDIRECTIO -DG__OSFDLL -DG__STD_EXCEPTION")
     SET(CINT_C_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__ROOT -DG__REDIRECTIO -DG__OSFDLL -DG__STD_EXCEPTION")
  
     SET(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -flat_namespace -single_module -undefined dynamic_lookup")
     SET(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -flat_namespace -single_module -undefined dynamic_lookup")
  
     set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} -bind_at_load -m64")
     set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -bind_at_load -m64")

     # Select flags.
     set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
     set(CMAKE_CXX_FLAGS_RELEASE        "-O2")
     set(CMAKE_CXX_FLAGS_DEBUG          "-g -O2 -fno-reorder-blocks -fno-schedule-insns -fno-inline")
     set(CMAKE_CXX_FLAGS_DEBUGFULL      "-g3 -fno-inline")
     set(CMAKE_CXX_FLAGS_PROFILE        "-g3 -fno-inline -ftest-coverage -fprofile-arcs")
     set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g")
     set(CMAKE_C_FLAGS_RELEASE          "-O2")
     set(CMAKE_C_FLAGS_DEBUG            "-g -O2 -fno-reorder-blocks -fno-schedule-insns -fno-inline")
     set(CMAKE_C_FLAGS_DEBUGFULL        "-g3 -fno-inline")
     set(CMAKE_C_FLAGS_PROFILE          "-g3 -fno-inline -ftest-coverage -fprofile-arcs")
   
     #settings for cint
     set(CPPPREP "${CMAKE_CXX_COMPILER} -E -C")  
     set(CXXOUT "-o ")
     set(EXPLICITLINK "no") #TODO

     set(EXEEXT "")
     set(SOEXT "so")

  else (CMAKE_COMPILER_IS_GNUCXX)
    MESSAGE(FATAL_ERROR "There is no setup for this compiler up to now. Don't know waht to do. Stop cmake at this point.")
  endif (CMAKE_COMPILER_IS_GNUCXX)
  
else (CMAKE_SYSTEM_NAME MATCHES Darwin)
  MESSAGE(FATAL_ERROR "There is no setup for this this Apple system up to now. Don't know waht to do. Stop cmake at this point.")
endif (CMAKE_SYSTEM_NAME MATCHES Darwin)
