Set(ROOT_ARCHITECTURE linux)
Set(ROOT_PLATFORM linux)

execute_process(COMMAND uname -m OUTPUT_VARIABLE SYSCTL_OUTPUT)
    
If(${SYSCTL_OUTPUT} MATCHES x86_64)
  Message(STATUS "Found a 64bit system")
  Set(BIT_ENVIRONMENT "-m64")
  Set(SPECIAL_CINT_FLAGS "-DG__64BIT")
  If(CMAKE_COMPILER_IS_GNUCXX)
    Message(STATUS "Found GNU compiler collection")
    Set(ROOT_ARCHITECTURE linuxx8664gcc)
  Else(CMAKE_COMPILER_IS_GNUCXX)
    If(${CMAKE_CXX_COMPILER} MATCHES "icpc")
      Set(ROOT_ARCHITECTURE linuxx8664icc)
    Else(${CMAKE_CXX_COMPILER} MATCHES "icpc")
      Message(FATAL_ERROR "There is no Setup for this compiler up to now. Don't know what to do. Stop cmake at this point.")
    EndIf(${CMAKE_CXX_COMPILER} MATCHES "icpc")
  EndIf(CMAKE_COMPILER_IS_GNUCXX)
Else(${SYSCTL_OUTPUT} MATCHES x86_64)
  Message(STATUS "Found a 32bit system")
  Set(BIT_ENVIRONMENT "-m32")
  Set(SPECIAL_CINT_FLAGS "")
  If(CMAKE_COMPILER_IS_GNUCXX)
    Message(STATUS "Found GNU compiler collection")
    Set(ROOT_ARCHITECTURE linux)
  Else(CMAKE_COMPILER_IS_GNUCXX)
    If(${CMAKE_CXX_COMPILER} MATCHES "icpc")
      Set(ROOT_ARCHITECTURE linuxicc)
    Else(${CMAKE_CXX_COMPILER} MATCHES "icpc")
      Message(FATAL_ERROR "There is no Setup for this compiler up to now. Don't know what to do. Stop cmake at this point.")
    EndIf(${CMAKE_CXX_COMPILER} MATCHES "icpc")
  EndIf(CMAKE_COMPILER_IS_GNUCXX)
EndIf(${SYSCTL_OUTPUT} MATCHES x86_64)

If(CMAKE_COMPILER_IS_GNUCXX)

  Set(SYSLIBS "-lm -ldl ${CMAKE_THREAD_LIBS_INIT} -rdynamic") 
  Set(XLIBS "${XPMLIBDIR} ${XPMLIB} ${X11LIBDIR} -lXext -lX11")
  Set(CILIBS "-lm -ldl -rdynamic")
  Set(CRYPTLIBS "-lcrypt")


   Set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe ${BIT_ENVIRONMENT} -Wall -W -Woverloaded-virtual -fPIC")
   Set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe ${BIT_ENVIRONMENT} -Wall -W -fPIC")

   Set(CMAKE_Fortran_FLAGS "${CMAKE_FORTRAN_FLAGS} ${BIT_ENVIRONMENT}")

   Set(CINT_CXX_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__OSFDLL -DG__ROOT -DG__REDIRECTIO -DG__STD_EXCEPTION ${SPECIAL_CINT_FLAGS}")
   Set(CINT_C_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__OSFDLL -DG__ROOT -DG__REDIRECTIO -DG__STD_EXCEPTION ${SPECIAL_CINT_FLAGS}")

   Set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS}")
   Set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS}")

   Set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} ${BIT_ENVIRONMENT} -ldl")
   Set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${BIT_ENVIRONMENT} -ldl")

   # Select flags.
   Set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
   Set(CMAKE_CXX_FLAGS_RELEASE        "-O2")
   Set(CMAKE_CXX_FLAGS_DEBUG          "-g  -fno-reorder-blocks -fno-schedule-insns -fno-inline")
   Set(CMAKE_CXX_FLAGS_DEBUGFULL      "-g3 -fno-inline")
   Set(CMAKE_CXX_FLAGS_PROFILE        "-g3 -fno-inline -ftest-coverage -fprofile-arcs")
   Set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g")
   Set(CMAKE_C_FLAGS_RELEASE          "-O2")
   Set(CMAKE_C_FLAGS_DEBUG            "-g  -fno-reorder-blocks -fno-schedule-insns -fno-inline")
   Set(CMAKE_C_FLAGS_DEBUGFULL        "-g3 -fno-inline")
   Set(CMAKE_C_FLAGS_PROFILE          "-g3 -fno-inline -ftest-coverage -fprofile-arcs")
 
   #Settings for cint
   Set(CPPPREP "${CMAKE_CXX_COMPILER} -E -C")  
   Set(CXXOUT "-o ")
   Set(EXPLICITLINK "no") #TODO

   Set(EXEEXT "")
   Set(SOEXT "so")
Else(CMAKE_COMPILER_IS_GNUCXX)
  If(${CMAKE_CXX_COMPILER} MATCHES "icpc")

    Set(SYSLIBS "-limf -lm -ldl ${CMAKE_THREAD_LIBS_INIT} -rdynamic") 
    Set(XLIBS "${XPMLIBDIR} ${XPMLIB} ${X11LIBDIR} -lXext -lX11")
    Set(CILIBS "-limf -lm -ldl -rdynamic")
    Set(CRYPTLIBS "-lcrypt")
  
  
     Set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -wd1476")
     Set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -restrict")
  
     Set(CMAKE_Fortran_FLAGS "${CMAKE_FORTRAN_FLAGS} -fPIC")
  
     Set(CINT_CXX_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__OSFDLL -DG__ROOT -DG__REDIRECTIO -DG__STD_EXCEPTION ${SPECIAL_CINT_FLAGS}")
     Set(CINT_C_DEFINITIONS "-DG__REGEXP -DG__UNIX -DG__SHAREDLIB -DG__OSFDLL -DG__ROOT -DG__REDIRECTIO -DG__STD_EXCEPTION ${SPECIAL_CINT_FLAGS}")
  
  
 
   # Check icc compiler version and set compile flags according to the
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -v  ERROR_VARIABLE _icc_version_info)

    STRING(REGEX REPLACE "^Version[ ]([0-9]+)\\.[0-9]+" "\\1" ICC_MAJOR "${_icc_version_info}")
    STRING(REGEX REPLACE "^Version[ ][0-9]+\\.([0-9]+)" "\\1" ICC_MINOR "${_icc_version_info}")

    MESSAGE(STATUS "Found ICC major version ${ICC_MAJOR}")
    MESSAGE(STATUS "Found ICC minor version ${ICC_MINOR}")
  
    If(ICC_MAJOR EQUAL 9)  
      Set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd1572")
      Set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -wd1572")
      Set(ICC_GE_9  9)
    EndIf(ICC_MAJOR EQUAL 9)  
  
    If(ICC_MAJOR EQUAL 10)  
      Set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd1572")
      Set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -wd1572")
      Set(ICC_GE_9  10)
      If(ICC_MINOR GREATER 0)  
        Set(ICC_GE_101 101)
      EndIf(ICC_MINOR GREATER 0)  
    EndIf(ICC_MAJOR EQUAL 10)  
  
    If(ICC_MAJOR EQUAL 11)  
      Set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BIT_ENVIRONMENT} -wd1572 -wd279")
      Set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${BIT_ENVIRONMENT} -wd1572 -wd279")
      Set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} ${BIT_ENVIRONMENT}")
      Set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${BIT_ENVIRONMENT}")
      Set(ICC_GE_9  11)
      Set(ICC_GE_101 110)
    EndIf(ICC_MAJOR EQUAL 11)  
  
     Set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS}")
     Set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS}")
  
  
     # Select flags.
     Set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O0 -g")
     Set(CMAKE_CXX_FLAGS_RELEASE        "-O")
     Set(CMAKE_CXX_FLAGS_DEBUG          "-g -O")
     Set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O0 -g")
     Set(CMAKE_C_FLAGS_RELEASE          "-O")
     Set(CMAKE_C_FLAGS_DEBUG            "-g -O2")
   
     #Settings for cint
     Set(CPPPREP "${CMAKE_CXX_COMPILER} -E -C")  
     Set(CXXOUT "-o ")
     Set(EXPLICITLINK "no") #TODO
  
     Set(EXEEXT "")
     Set(SOEXT "so")
  Else(${CMAKE_CXX_COMPILER} MATCHES "icpc")

  EndIf(${CMAKE_CXX_COMPILER} MATCHES "icpc")
Endif(CMAKE_COMPILER_IS_GNUCXX)
  
#Set(ICC_MAJOR ${ICC_MAJOR} PARENT_SCOPE)
#Set(ICC_MAJOR ${ICC_MINOR} PARENT_SCOPE)
