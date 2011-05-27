set(ROOT_ARCHITECTURE win32)
set(ROOT_PLATFORM win32)

math(EXPR VC_MAJOR "${MSVC_VERSION} / 100")
math(EXPR VC_MINOR "${MSVC_VERSION} % 100")

set(SOEXT dll)
set(EXEEXT exe)

set(SYSLIBS advapi32.lib)
set(XLIBS)
set(CILIBS)
set(CRYPTLIBS)

#---Select compiler flags----------------------------------------------------------------
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -Z7")
set(CMAKE_CXX_FLAGS_RELEASE        "-O2")
set(CMAKE_CXX_FLAGS_DEBUG          "-Z7")
set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -Z7")
set(CMAKE_C_FLAGS_RELEASE          "-O2")
set(CMAKE_C_FLAGS_DEBUG            "-Z7")

set(CMAKE_CXX_FLAGS "-nologo -I${CMAKE_SOURCE_DIR}/build/win -FIw32pragma.h -FIsehmap.h -MD -GR -EHsc- -W3 -wd4244 -D_WIN32")
set(CMAKE_C_FLAGS   "-nologo -I${CMAKE_SOURCE_DIR}/build/win -FIw32pragma.h -FIsehmap.h -MD -EHsc- -W3 -D_WIN32")

#---Set Linker flags----------------------------------------------------------------------
#set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS})
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -ignore:4049,4206,4217,4221 -incremental:no")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -ignore:4049,4206,4217,4221 -incremental:no")


