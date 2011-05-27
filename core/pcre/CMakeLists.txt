# TODO: Check if we have to install the buildin pcre version
# or if we can use the system version of pcre. We have to create 
# a FindPCRE.cmake script and search for the system version of
# pcre if not set buildin version of pcre.
# Up to now we don't check and install the buildin version anyway.
##is_root_install_option_enabled(builtin_pcre)
##if(option_builtin_pcre_is_enabled)
###  MESSAGE("BLA: ${option_builtin_pcre_is_enabled}")
###  add_subdirectory (core/pcre)
##endif(option_builtin_pcre_is_enabled)

# This is not a verry clean solution, but the problem is that the pcre has its
# own tarfile and its own buildsystem. So we have to unpack the tarfile and
# then call the build system of pcre. The created library is imported into
# the scope of cmake, so even make clean works.

# Define package related variables 
set(PCRE_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(PCRE_VERSION "pcre-7.8")
set(PCRE_TARGZFILE ${PCRE_SRCDIR}/${PCRE_VERSION}.tar.gz)
set(PCRE_TARFILE ${PCRE_SRCDIR}/${PCRE_VERSION}.tar)

if(WIN32)
  set(pcreliba ${CMAKE_CURRENT_BINARY_DIR}/win32/libpcre-7.8.lib)
  set(pcrelib  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libpcre.lib)
  if(winrtdebug)  
    set(pcrebuild "libpcre - Win32 Debug")
  else()
    set(pcrebuild "libpcre - Win32 Release")
  endif()
  add_custom_command( OUTPUT ${pcreliba}
                      COMMAND cmake -E copy_directory ${PCRE_SRCDIR}/win32 win32
                      COMMAND cmake -E tar xvzf ${PCRE_TARGZFILE}
		              COMMAND cmake -E chdir win32 nmake -nologo -f Makefile.msc 
					                               CFG=${pcrebuild} 
												   NMCXXFLAGS="-I${CMAKE_SOURCE_DIR}/build/win -FIw32pragma.h")
else()
  set(pcreliba ${CMAKE_CURRENT_BINARY_DIR}/${PCRE_VERSION}/.libs/libpcre.a)
  set(pcrelib  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libpcre.a)
  # Define special compiler settings for prce
  set(PCRE_CC ${CMAKE_C_COMPILER})
  set(PCRE_CFLAGS "-O")
  if(ROOT_ARCHITECTURE MATCHES alphacxx6)
    set(PCRE_CC "cc")  
  elseif(ROOT_ARCHITECTURE MATCHES linuxx8664gcc)
    set(PCRE_CFLAGS "-m64 -O")
  elseif(ROOT_ARCHITECTURE MATCHES linuxx8664icc)
    set(PCRE_CFLAGS "-m64 -O")
  elseif(ROOT_ARCHITECTURE MATCHES linuxppc64gcc)
    set(PCRE_CFLAGS "-m64 -O")
  elseif(ROOT_ARCHITECTURE MATCHES linuxicc)
    set(PCRE_CFLAGS "-m32 -O")
  elseif(ROOT_ARCHITECTURE MATCHES linux)
    set(PCRE_CFLAGS "-m32 -O")
  elseif(ROOT_ARCHITECTURE MATCHES macosx64)
    set(PCRE_CFLAGS "-m64 -O")
  elseif(ROOT_ARCHITECTURE MATCHES macosx)
    set(PCRE_CFLAGS "-m32 -O")
  elseif(ROOT_ARCHITECTURE MATCHES solaris64CC5)
    set(PCRE_CFLAGS "-m64 -O")
  elseif(ROOT_ARCHITECTURE MATCHES sgicc64)
    set(PCRE_CC "cc")  
    set(PCRE_CFLAGS "-64 -O")
  elseif(ROOT_ARCHITECTURE MATCHES hpuxia64acc)
    set(PCRE_CC "cc")  
    set(PCRE_CFLAGS "+DD64 -Ae -O")
  endif()
    
  # copy tarfile from source directory to build directory, unpack the tarfile
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${PCRE_VERSION}/unpacked
                     COMMAND cmake -E copy ${PCRE_TARGZFILE} .
                     COMMAND gunzip ${PCRE_VERSION}.tar.gz
                     COMMAND tar xf ${PCRE_VERSION}.tar
                     COMMAND touch ${PCRE_VERSION}/unpacked
                     COMMAND cmake -E remove -f ${PCRE_VERSION}.tar
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )
  # configure and make pcre, this command depends on the unpacked package 
  add_custom_command(OUTPUT ${pcreliba}  
                     COMMAND CC=${PCRE_CC} CFLAGS=${PCRE_CFLAGS} ./configure --with-pic --disable-shared > /dev/null 2>& 1
                     COMMAND make libpcre.la > /dev/null 2>& 1
                     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PCRE_VERSION} 
                     DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${PCRE_VERSION}/unpacked )
endif()

# copy the created library into the library directory in the build directory
# This command depends on the created libpcre.a
if(ROOT_PLATFORM MATCHES macosx)
  add_custom_command(OUTPUT ${pcrelib}
                     COMMAND cmake -E copy_if_different ${pcreliba} ${pcrelib}
                     COMMAND ranlib ${pcrelib}
                     DEPENDS ${pcreliba} )
else()
  add_custom_command(OUTPUT ${pcrelib}
                     COMMAND cmake -E copy_if_different ${pcreliba} ${pcrelib}
                     DEPENDS ${pcreliba} )
endif()
                
# create a target which will always be build and does actually nothing. The target is only
# needed that the dependencies are build, if they are not up to date. If everything is up to
# dte nothing is done. This target depends on the libpcre.a in the library directory of the 
# build directory.
add_custom_target(PCRE DEPENDS ${pcrelib})
set_target_properties(PCRE PROPERTIES FOLDER Builtins)

