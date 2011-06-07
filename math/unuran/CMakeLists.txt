############################################################################
# CMakeLists.txt file for building ROOT math/unurun package
############################################################################

#---Define package related variables----------------------------------------------------------------- 
set(UNR_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(UNR_VERSION "1.8.0-root")
set(UNR_TARNAME "unuran-${UNR_VERSION}")
set(UNR_TARGZFILE ${UNR_SRCDIR}/${UNR_TARNAME}.tar.gz)
set(UNR_TARFILE ${UNR_SRCDIR}/${UNR_TARNAME}.tar)
set(UNR_UNTARDIR ${CMAKE_CURRENT_BINARY_DIR}/${UNR_TARNAME})

#---Untar sources at configuration/generation time (needed for listing sources)
if(NOT EXISTS ${UNR_UNTARDIR})
  execute_process( COMMAND cmake -E tar xzf ${UNR_TARGZFILE}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} )
endif()

if(WIN32)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.win.in  ${UNR_UNTARDIR}/config.h)
else()  
  #---Define special compiler settings for unurun-----------------------------------------------------
  set(UNR_CC ${CMAKE_C_COMPILER})
  if(ROOT_ARCHITECTURE MATCHES sgicc64)
    set(UNR_CC "gcc -mabi=64")
  elseif(ROOT_ARCHITECTURE MATCHES hpuxia64acc)
    set(UNR_CC "cc +DD64 -Ae")  
  elseif(ROOT_ARCHITECTURE MATCHES linuxppc64gcc)
    set(UNR_CC "gcc -m64 -fPIC")
  elseif(ROOT_ARCHITECTURE MATCHES linuxx8664gcc)
    set(UNR_CC "gcc")
    set(UNR_CFLAGS "-m64 -fPIC")
  elseif(ROOT_ARCHITECTURE MATCHES linuxicc)
    set(UNR_CC "icc")
    set(UNR_CFLAGS "-m32")
  elseif(ROOT_ARCHITECTURE MATCHES linuxx8664icc)
    set(UNR_CC "icc")
    set(UNR_CFLAGS "-m64")
  endif()

  #---configure unuran (required for creating the config.h used by unuran source files)----------------
  add_custom_command(OUTPUT ${UNR_UNTARDIR}/config.h  
                     COMMAND GNUMAKE=make ./configure CC=${UNR_CC} CFLAGS=${UNR_CFLAGS}
                     WORKING_DIRECTORY ${UNR_UNTARDIR})
endif()

ROOT_USE_PACKAGE(hist/hist)

include_directories(BEFORE ${UNR_UNTARDIR} ${UNR_UNTARDIR}/src ${UNR_UNTARDIR}/src/utils)
add_definitions(-DHAVE_CONFIG_H)
set(unrsources ${UNR_UNTARDIR}/src/utils/*.c
               ${UNR_UNTARDIR}/src/methods/*.c
               ${UNR_UNTARDIR}/src/specfunct/*.c
               ${UNR_UNTARDIR}/src/distr/*.c
               ${UNR_UNTARDIR}/src/distributions/*.c
               ${UNR_UNTARDIR}/src/parser/*.c
               ${UNR_UNTARDIR}/src/tests/*.c
               ${UNR_UNTARDIR}/src/uniform/*.c
               ${UNR_UNTARDIR}/src/urng/*.c )

ROOT_GENERATE_DICTIONARY(G__Unuran *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Unuran LINKDEF LinkDef.h DEPENDENCIES Hist MathCore)
ROOT_LINKER_LIBRARY(Unuran *.cxx ${unrsources} G__Unuran.cxx ${UNR_UNTARDIR}/config.h  LIBRARIES Cint Core DEPENDENCIES Hist MathCore)
ROOT_INSTALL_HEADERS()
