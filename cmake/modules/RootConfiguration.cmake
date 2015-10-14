INCLUDE (CheckCXXSourceCompiles)

#---Define a function to do not polute the top level namespace with unneeded variables-----------------------
function(RootConfigure)

#---Define all sort of variables to bridge between the old Module.mk and the new CMake equivalents-----------
foreach(v 1 ON YES TRUE Y on yes true y)
  set(value${v} yes)
endforeach()
foreach(v 0 OFF NO FALSE N IGNORE off no false n ignore)
  set(value${v} no)
endforeach()

set(ROOT_DICTTYPE cint)
#set(ROOT_CONFIGARGS "")
set(top_srcdir ${CMAKE_SOURCE_DIR})
set(top_builddir ${CMAKE_BINARY_DIR})
set(architecture ${ROOT_ARCHITECTURE})
set(platform ${ROOT_PLATFORM})
set(host)
set(useconfig FALSE)
set(major ${ROOT_MAJOR_VERSION})
set(minor ${ROOT_MINOR_VERSION})
set(revis ${ROOT_PATCH_VERSION})
set(mkliboption "-v ${major} ${minor} ${revis} ")
set(cflags ${CMAKE_CXX_FLAGS})
set(ldflags ${CMAKE_CXX_LINK_FLAGS})

set(winrtdebug ${value${winrtdebug}})
set(exceptions ${value${exceptions}})
set(explicitlink ${value${explicitlink}})

if(gnuinstall)
  set(prefix ${CMAKE_INSTALL_PREFIX})
else()
  set(prefix $(ROOTSYS))
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_SYSCONFDIR})
  set(etcdir ${CMAKE_INSTALL_SYSCONFDIR})
else()
  set(etcdir ${prefix}/${CMAKE_INSTALL_SYSCONFDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_BINDIR})
  set(bindir ${CMAKE_INSTALL_BINDIR})
else()
  set(bindir ${prefix}/${CMAKE_INSTALL_BINDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_LIBDIR})
  set(libdir ${CMAKE_INSTALL_LIBDIR})
else()
  set(libdir ${prefix}/${CMAKE_INSTALL_LIBDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_INCLUDEDIR})
  set(incdir ${CMAKE_INSTALL_INCLUDEDIR})
else()
  set(incdir ${prefix}/${CMAKE_INSTALL_INCLUDEDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_MANDIR})
  set(mandir ${CMAKE_INSTALL_MANDIR})
else()
  set(mandir ${prefix}/${CMAKE_INSTALL_MANDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_SYSCONFDIR})
  set(plugindir ${CMAKE_INSTALL_SYSCONFDIR}/plugins)
else()
  set(plugindir ${prefix}/${CMAKE_INSTALL_SYSCONFDIR}/plugins)
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_DATADIR})
  set(datadir ${CMAKE_INSTALL_DATADIR})
else()
  set(datadir ${prefix}/${CMAKE_INSTALL_DATADIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_ELISPDIR})
  set(elispdir ${CMAKE_INSTALL_ELISPDIR})
else()
  set(elispdir ${prefix}/${CMAKE_INSTALL_ELISPDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_FONTDIR})
  set(ttffontdir ${CMAKE_INSTALL_FONTDIR})
else()
  set(ttffontdir ${prefix}/${CMAKE_INSTALL_FONTDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_MACRODIR})
  set(macrodir ${CMAKE_INSTALL_MACRODIR})
else()
  set(macrodir ${prefix}/${CMAKE_INSTALL_MACRODIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_SRCDIR})
  set(srcdir ${CMAKE_INSTALL_SRCDIR})
else()
  set(srcdir ${prefix}/${CMAKE_INSTALL_SRCDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_ICONDIR})
  set(iconpath ${CMAKE_INSTALL_ICONDIR})
else()
  set(iconpath ${prefix}/${CMAKE_INSTALL_ICONDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_CINTINCDIR})
  set(cintincdir ${CMAKE_INSTALL_CINTINCDIR})
else()
  set(cintincdir ${prefix}/${CMAKE_INSTALL_CINTINCDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_DOCDIR})
  set(docdir ${CMAKE_INSTALL_DOCDIR})
else()
  set(docdir ${prefix}/${CMAKE_INSTALL_DOCDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_TESTDIR})
  set(testdir ${CMAKE_INSTALL_TESTDIR})
else()
  set(testdir ${prefix}/${CMAKE_INSTALL_TESTDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_TUTDIR})
  set(tutdir ${CMAKE_INSTALL_TUTDIR})
else()
  set(tutdir ${prefix}/${CMAKE_INSTALL_TUTDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_ACLOCALDIR})
  set(aclocaldir ${CMAKE_INSTALL_ACLOCALDIR})
else()
  set(aclocaldir ${prefix}/${CMAKE_INSTALL_ACLOCALDIR})
endif()

set(LibSuffix ${SOEXT})

set(buildx11 ${value${x11}})
set(x11libdir -L${X11_LIBRARY_DIR})
set(xpmlibdir -L${X11_LIBRARY_DIR})
set(xpmlib ${X11_Xpm_LIB})
set(enable_xft ${value${xft}})

set(enable_thread ${value${thread}})
set(threadflag ${CMAKE_THREAD_FLAG})
set(threadlibdir)
set(threadlib ${CMAKE_THREAD_LIBS_INIT})

set(builtinfreetype ${value${builtin_freetype}})
set(builtinpcre ${value${builtin_pcre}})

set(builtinzlib ${value${builtin_zlib}})
set(zliblibdir ${ZLIB_LIBRARY_DIR})
set(zliblib ${ZLIB_LIBRARY})
set(zlibincdir ${ZLIB_INCLUDE_DIR})

set(buildgl ${value${opengl}})
set(opengllibdir ${OPENGL_LIBRARY_DIR})
set(openglulib ${OPENGL_glu_LIBRARY})
set(opengllib ${OPENGL_gl_LIBRARY})
set(openglincdir ${OPENGL_INCLUDE_DIR})

set(buildldap ${value${ldap}})
set(ldaplibdir ${LDAP_LIBRARY_DIR})
set(ldaplib ${LDAP_LIBRARY})
set(ldapincdir ${LDAP_INCLUDE_DIR})

set(buildmysql ${value${mysql}})
set(mysqllibdir ${MYSQL_LIBRARY_DIR})
set(mysqllib ${MYSQL_LIBRARY})
set(mysqlincdir ${MYSQL_INCLUDE_DIR})

set(buildoracle ${value${oracle}})
set(oraclelibdir ${ORACLE_LIBRARY_DIR})
set(oraclelib ${ORACLE_LIBRARY})
set(oracleincdir ${ORACLE_INCLUDE_DIR})

set(buildpgsql ${value${pgsql}})
set(pgsqllibdir ${PGSQL_LIBRARY_DIR})
set(pgsqllib ${PGSQL_LIBRARY})
set(pgsqlincdir ${PGSQL_INCLUDE_DIR})

set(buildsqlite ${value${sqlite}})
set(sqlitelibdir ${SQLITE_LIBRARY_DIR})
set(sqlitelib ${SQLITE_LIBRARY})
set(sqliteincdir ${SQLITE_INCLUDE_DIR})

set(buildsapdb ${value${sapdb}})
set(sapdblibdir ${SAPDB_LIBRARY_DIR})
set(sapdblib ${SAPDB_LIBRARY})
set(sapdbincdir ${SAPDB_INCLUDE_DIR})

set(buildodbc ${value${odbc}})
set(odbclibdir ${OCDB_LIBRARY_DIR})
set(odbclib ${OCDB_LIBRARY})
set(odbcincdir ${OCDB_INCLUDE_DIR})

set(buildqt ${value${qt}})
set(buildqtpsi ${value${qtgsi}})
set(qtlibdir ${QT_LIBRARY_DIR})
set(qtlib ${QT_QT_LIBRARY})
set(qtincdir ${QT_INCLUDE_DIR})
set(qtvers ${QT_VERSION_MAJOR})
set(qtmocexe ${QT_MOC_EXECUTABLE})

set(buildrfio ${value${rfio}})
set(shiftlibdir ${RFIO_LIBRARY_DIR})
set(shiftlib ${RFIO_LIBRARY})
set(shiftincdir ${RFIO_INCLUDE_DIR})
set(shiftcflags)

set(buildcastor ${value${castor}})
set(castorlibdir ${CASTOR_LIBRARY_DIR})
set(castorlib ${CASTOR_LIBRARY})
set(castorincdir ${CASTOR_INCLUDE_DIR})
set(castorcflags)


set(builddavix ${value${davix}})
set(davixlibdir ${DAVIX_LIBRARY_DIR})
set(davixlib ${DAVIX_LIBRARY})
set(davixincdir ${DAVIX_INCLUDE_DIR})
if(davix)
  set(useoldwebfile no)
else()
  set(useoldwebfile yes)
endif()

set(buildnetxng ${value${netxng}})
if(netxng)
  set(useoldnetx no)
else()
  set(useoldnetx yes)
endif()

set(builddcap ${value${dcap}})
set(dcaplibdir ${DCAP_LIBRARY_DIR})
set(dcaplib ${DCAP_LIBRARY})
set(dcapincdir ${DCAP_INCLUDE_DIR})

set(buildftgl ${value${builtin_ftgl}})
set(ftgllibdir ${FTGL_LIBRARY_DIR})
set(ftgllibs ${FTGL_LIBRARIES})
set(ftglincdir ${FTGL_INCLUDE_DIR})

set(buildglew ${value${builtin_glew}})
set(glewlibdir ${GLEW_LIBRARY_DIR})
set(glewlibs ${GLEW_LIBRARIES})
set(glewincdir ${GLEW_INCLUDE_DIR})

set(buildgfal ${value${gfal}})
set(gfallibdir ${GFAL_LIBRARY_DIR})
set(gfallib ${GFAL_LIBRARY})
set(gfalincdir ${GFAL_INCLUDE_DIR})

set(buildglite ${value${glite}})
set(glitelibdir ${GLITE_LIBRARY_DIR})
set(glitelib ${GLITE_LIBRARY})
set(gaw_cppflags)

set(buildmemstat ${value${memstat}})

set(buildbonjour ${value${bonjour}})
set(dnssdlibdir ${BONJOUR_LIBRARY_DIR})
set(dnssdlib ${BONJOUR_LIBRARY})
set(dnsdincdir ${BONJOUR_INCLUDE_DIR})
set(bonjourcppflags)

set(buildchirp ${value${chirp}})
set(chirplibdir ${CHIRP_LIBRARY_DIR})
set(chirplib ${CHIRP_LIBRARY})
set(chirpincdir ${CHIRP_INCLUDE_DIR})

set(buildhdfs ${value${hdfs}})
set(hdfslibdir ${HDFS_LIBRARY_DIR})
set(hdfslib ${HDFS_LIBRARY})
set(hdfsincdir ${HDFS_INCLUDE_DIR})

set(jniincdir ${Java_INCLUDE_DIRS})
set(jvmlib ${Java_LIBRARIES})
set(jvmlibdir ${Java_LIBRARY_DIR})

set(buildalien ${value${alien}})
set(alienlibdir ${ALIEN_LIBRARY_DIR})
set(alienlib ${ALIEN_LIBRARY})
set(alienincdir ${ALIEN_INCLUDE_DIR})

set(buildasimage ${value${asimage}})
set(builtinafterimage ${builtin_afterimage})
set(asextralib ${ASEXTRA_LIBRARIES})
set(asextralibdir)
set(asjpegincdir ${JPEG_INCLUDE_DIR})
set(aspngincdir ${PNG_INCLUDE_DIR})
set(astiffincdir ${TIFF_INCLUDE_DIR})
set(asgifincdir ${GIF_INCLUDE_DIR})
set(asimageincdir)
set(asimagelib)
set(asimagelibdir)

set(buildpythia6 ${value${pythia6}})
set(pythia6libdir ${PYTHIA6_LIBRARY_DIR})
set(pythia6lib ${PYTHIA6_LIBRARY})
set(pythia6cppflags)
set(buildpythia8 ${value${pythia8}})
set(pythia8libdir ${PYTHIA8_LIBRARY_DIR})
set(pythia8lib ${PYTHIA8_LIBRARY})
set(pythia8cppflags)

set(buildfftw3 ${value${fftw3}})
set(fftw3libdir ${FFTW3_LIBRARY_DIR})
set(fftw3lib ${FFTW3_LIBRARY})
set(fftw3incdir ${FFTW3_INCLUDE_DIR})

set(buildfitsio ${value${fitsio}})
set(fitsiolibdir ${FITSIO_LIBRARY_DIR})
set(fitsiolib ${FITSIO_LIBRARY})
set(fitsioincdir ${FITSIO_INCLUDE_DIR})

set(buildgviz ${value${gviz}})
set(gvizlibdir ${GVIZ_LIBRARY_DIR})
set(gvizlib ${GVIZ_LIBRARY})
set(gvizincdir ${GVIZ_INCLUDE_DIR})
set(gvizcflags)

set(buildpython ${value${python}})
set(pythonlibdir ${PYTHON_LIBRARY_DIR})
set(pythonlib ${PYTHON_LIBRARY})
set(pythonincdir ${PYTHON_INCLUDE_DIR})
set(pythonlibflags)

set(buildruby ${value${ruby}})
set(rubylibdir ${RUBY_LIBRARY_DIR})
set(rubylib ${RUBY_LIBRARY})
set(rubyincdir ${RUBY_INCLUDE_DIR})

set(buildxml ${value${xml}})
set(xmllibdir ${LIBXML2_LIBRARY_DIR})
set(xmllib ${LIBXML2_LIBRARIES})
set(xmlincdir ${LIBXML2_INCLUDE_DIR})

set(buildxrd ${value${xrootd}})
set(xrdlibdir )
set(xrdincdir)
set(xrdaddopts)
set(extraxrdflags)
set(xrdversion)

set(srplibdir)
set(srplib)
set(srpincdir)

set(buildsrputil)
set(srputillibdir)
set(srputillib)
set(srputilincdir)

set(afslib ${AFS_LIBRARY})
set(afslibdir ${AFS_LIBRARY_DIR})
set(afsincdir ${AFS_INCLUDE_DIR})
set(afsextracflags)
set(afsshared)

set(alloclib)
set(alloclibdir)

set(buildkrb5 ${value${krb5}})
set(krb5libdir ${KRB5_LIBRARY_DIR})
set(krb5lib ${KRB5_LIBRARY})
set(krb5incdir ${KRB5_INCLUDE_DIR})
set(krb5init ${KRB5_INIT})

set(comerrlib)
set(comerrlibdir)
set(resolvlib)
set(cryptolib ${CRYPTLIBS})
set(cryptolibdir)

set(buildglobus ${value${globus}})
set(globuslibdir ${GLOBUS_LIBRARY_DIR})
set(globuslib ${GLOBUS_LIBRARY})
set(globusincdir ${GLOBUS_INCLUDE_DIR})
set(buildxrdgsi)

set(buildmonalisa ${value${monalisa}})
set(monalisalibdir ${MONALISA_LIBRARY_DIR})
set(monalisalib ${MONALISA_LIBRARY})
set(monalisaincdir ${MONALISA_INCLUDE_DIR})

set(ssllib ${OPENSSL_LIBRARIES})
set(ssllibdir)
set(sslincdir ${OPENSSL_INCLUDE_DIR})
set(sslshared)

set(gsllibs ${GSL_LIBRARIES})
set(gsllibdir)
set(gslincdir ${GSL_INCLUDE_DIR})
set(gslflags)

set(shadowpw ${value${shadowpw}})
set(buildgenvector ${value${genvector}})
set(buildmathmore ${value${mathmore}})
set(buildcling ${value${cling}})
set(buildroofit ${value${roofit}})
set(buildminuit2 ${value${minuit2}})
set(buildunuran ${value${unuran}})
set(buildgdml ${value${gdml}})
set(buildhttp ${value${http}})
set(buildtable ${value${table}})
set(buildtmva ${value${tmva}})

set(cursesincdir ${CURSES_INCLUDE_DIR})
set(curseslibdir)
set(curseslib ${CURSES_LIBRARIES})
set(curseshdr ${CURSES_HEADER_FILE})
set(buildeditline ${value${editline}})
set(cppunit)
set(dicttype ${ROOT_DICTTYPE})

#---RConfigure-------------------------------------------------------------------------------------------------
set(setresuid undef)
if(mathmore)
  set(hasmathmore define)
else()
  set(hasmathmore undef)
endif()
if(CMAKE_USE_PTHREADS_INIT)
  set(haspthread define)
else()
  set(haspthread undef)
endif()
if(xft)
  set(hasxft define)
else()
  set(hasxft undef)
endif()
if(cling)
  set(hascling define)
else()
  set(hascling undef)
endif()
if(lzma)
  set(haslzmacompression define)
else()
  set(haslzmacompression undef)
endif()
if(cocoa)
  set(hascocoa define)
else()
  set(hascocoa undef)
endif()
if(vc)
  set(hasvc define)
else()
  set(hasvc undef)
endif()
if(cxx11)
  set(cxxversion cxx11)
  set(usec++11 define)
else()
  set(usec++11 undef)
endif()
if(cxx14)
  set(cxxversion cxx14)
  set(usec++14 define)
else()
  set(usec++14 undef)
endif()
if(libcxx)
  set(uselibc++ define)
else()
  set(uselibc++ undef)
endif()
set(hasllvm undef)
set(llvmdir /**/)
if(gcctoolchain)
  set(setgcctoolchain define)
else()
  set(setgcctoolchain undef)
endif()

CHECK_CXX_SOURCE_COMPILES("#include <string_view>
  int main() { std::string_view().to_string(); return 0;}" found_stdstringview)
if(found_stdstringview)
  set(hasstdstringview define)
else()
  set(hasstdstringview undef)
endif()

CHECK_CXX_SOURCE_COMPILES("#include <experimental/string_view>
   int main() { std::experimental::string_view().to_string(); return 0;}" found_stdexpstringview)
if(found_stdexpstringview)
  set(hasstdexpstringview define)
else()
  set(hasstdexpstringview undef)
endif()

#---root-config----------------------------------------------------------------------------------------------
ROOT_SHOW_OPTIONS(features)
string(REPLACE "c++11" "cxx11" features ${features}) # change the name of the c++11 feature needed for root-config.in
set(configfeatures ${features})
set(configargs ${ROOT_CONFIGARGS})
set(configoptions ${ROOT_CONFIGARGS})
get_filename_component(altcc ${CMAKE_C_COMPILER} NAME)
get_filename_component(altcxx ${CMAKE_CXX_COMPILER} NAME)
get_filename_component(altf77 "${CMAKE_Fortran_COMPILER}" NAME)
get_filename_component(altld ${CMAKE_CXX_COMPILER} NAME)

set(pythonvers ${PYTHON_VERSION})

#---RConfigure.h---------------------------------------------------------------------------------------------
configure_file(${PROJECT_SOURCE_DIR}/config/RConfigure.in include/RConfigure.h NEWLINE_STYLE UNIX)
install(FILES ${CMAKE_BINARY_DIR}/include/RConfigure.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

#---Configure and install various files----------------------------------------------------------------------
execute_Process(COMMAND hostname OUTPUT_VARIABLE BuildNodeInfo OUTPUT_STRIP_TRAILING_WHITESPACE )

configure_file(${CMAKE_SOURCE_DIR}/config/rootrc.in ${CMAKE_BINARY_DIR}/etc/system.rootrc @ONLY NEWLINE_STYLE UNIX)
configure_file(${CMAKE_SOURCE_DIR}/config/RConfigOptions.in include/RConfigOptions.h NEWLINE_STYLE UNIX)
if(ruby)
  file(APPEND ${CMAKE_BINARY_DIR}/include/RConfigOptions.h "\#define R__RUBY_MAJOR ${RUBY_MAJOR_VERSION}\n\#define R__RUBY_MINOR ${RUBY_MINOR_VERSION}\n")
endif()

configure_file(${CMAKE_SOURCE_DIR}/config/Makefile-comp.in config/Makefile.comp NEWLINE_STYLE UNIX)
configure_file(${CMAKE_SOURCE_DIR}/config/Makefile.in config/Makefile.config NEWLINE_STYLE UNIX)
configure_file(${CMAKE_SOURCE_DIR}/config/mimes.unix.in ${CMAKE_BINARY_DIR}/etc/root.mimes NEWLINE_STYLE UNIX)

#---Generate the ROOTConfig files to be used by CMake projects-----------------------------------------------
ROOT_SHOW_OPTIONS(ROOT_ENABLED_OPTIONS)
configure_file(${CMAKE_SOURCE_DIR}/cmake/scripts/ROOTConfig-version.cmake.in
               ${CMAKE_BINARY_DIR}/ROOTConfig-version.cmake @ONLY NEWLINE_STYLE UNIX)
configure_file(${CMAKE_SOURCE_DIR}/cmake/scripts/RootUseFile.cmake.in
               ${CMAKE_BINARY_DIR}/ROOTUseFile.cmake @ONLY NEWLINE_STYLE UNIX)

#---Compiler flags (because user apps are a bit dependent on them...)----------------------------------------
string(REGEX REPLACE "(^|[ ]*)-W[^ ]*" "" __cxxflags "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "(^|[ ]*)-W[^ ]*" "" __cflags "${CMAKE_C_FLAGS}")
string(REGEX REPLACE "(^|[ ]*)-W[^ ]*" "" __fflags "${CMAKE_fortran_FLAGS}")
set(ROOT_COMPILER_FLAG_HINTS "#
set(ROOT_CXX_FLAGS \"${__cxxflags}\")
set(ROOT_C_FLAGS \"${__cflags}\")
set(ROOT_fortran_FLAGS \"${__fflags}\")
set(ROOT_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS}\")")

#---To be used from the binary tree--------------------------------------------------------------------------
set(ROOT_INCLUDE_DIR_SETUP "
# ROOT configured for use from the build tree - absolute paths are used.
set(ROOT_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/include)
")
set(ROOT_LIBRARY_DIR_SETUP "
# ROOT configured for use from the build tree - absolute paths are used.
set(ROOT_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib)
")
set(ROOT_BINARY_DIR_SETUP "
# ROOT configured for use from the build tree - absolute paths are used.
set(ROOT_BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
")
set(ROOT_MODULE_PATH_SETUP "
# ROOT configured for use CMake modules from source tree
set(CMAKE_MODULE_PATH \${CMAKE_MODULE_PATH} ${CMAKE_MODULE_PATH})
")

get_property(exported_targets GLOBAL PROPERTY ROOT_EXPORTED_TARGETS)
export(TARGETS ${exported_targets} FILE ${PROJECT_BINARY_DIR}/ROOTConfig-targets.cmake)
configure_file(${CMAKE_SOURCE_DIR}/cmake/scripts/ROOTConfig.cmake.in
               ${CMAKE_BINARY_DIR}/ROOTConfig.cmake @ONLY NEWLINE_STYLE UNIX)

#---To be used from the install tree--------------------------------------------------------------------------
# Need to calculate actual relative paths from CMAKEDIR to other locations
file(RELATIVE_PATH ROOT_CMAKE_TO_INCLUDE_DIR "${CMAKE_INSTALL_FULL_CMAKEDIR}" "${CMAKE_INSTALL_FULL_INCLUDEDIR}")
file(RELATIVE_PATH ROOT_CMAKE_TO_LIB_DIR "${CMAKE_INSTALL_FULL_CMAKEDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")
file(RELATIVE_PATH ROOT_CMAKE_TO_BIN_DIR "${CMAKE_INSTALL_FULL_CMAKEDIR}" "${CMAKE_INSTALL_FULL_BINDIR}")

set(ROOT_INCLUDE_DIR_SETUP "
# ROOT configured for the install with relative paths, so use these
get_filename_component(ROOT_INCLUDE_DIRS \"\${_thisdir}/${ROOT_CMAKE_TO_INCLUDE_DIR}\" ABSOLUTE)
")
set(ROOT_LIBRARY_DIR_SETUP "
# ROOT configured for the install with relative paths, so use these
get_filename_component(ROOT_LIBRARY_DIR \"\${_thisdir}/${ROOT_CMAKE_TO_LIB_DIR}\" ABSOLUTE)
")
set(ROOT_BINARY_DIR_SETUP "
# ROOT configured for the install with relative paths, so use these
get_filename_component(ROOT_BINARY_DIR \"\${_thisdir}/${ROOT_CMAKE_TO_BIN_DIR}\" ABSOLUTE)
")
set(ROOT_MODULE_PATH_SETUP "
# ROOT configured for use CMake modules from installation tree
set(CMAKE_MODULE_PATH \${CMAKE_MODULE_PATH}  \${_thisdir}/modules)
")
configure_file(${CMAKE_SOURCE_DIR}/cmake/scripts/ROOTConfig.cmake.in
               ${CMAKE_BINARY_DIR}/installtree/ROOTConfig.cmake @ONLY NEWLINE_STYLE UNIX)
install(FILES ${CMAKE_BINARY_DIR}/ROOTConfig-version.cmake
              ${CMAKE_BINARY_DIR}/ROOTUseFile.cmake
              ${CMAKE_BINARY_DIR}/installtree/ROOTConfig.cmake DESTINATION ${CMAKE_INSTALL_CMAKEDIR})
install(EXPORT ${CMAKE_PROJECT_NAME}Exports FILE ROOTConfig-targets.cmake DESTINATION ${CMAKE_INSTALL_CMAKEDIR})


#---Especial definitions for root-config et al.--------------------------------------------------------------
if(prefix STREQUAL "$(ROOTSYS)")
  foreach(d prefix bindir libdir incdir etcdir mandir)
    string(REPLACE "$(ROOTSYS)" "$ROOTSYS"  ${d} ${${d}})
  endforeach()
endif()


#---compiledata.h--------------------------------------------------------------------------------------------
if(WIN32)
  # We cannot use the compiledata.sh script for windows
  configure_file(${CMAKE_SOURCE_DIR}/cmake/scripts/compiledata.win32.in include/compiledata.h NEWLINE_STYLE UNIX)
else()
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/build/unix/compiledata.sh include/compiledata.h "${CXX}"
        "${CMAKE_CXX_FLAGS_RELEASE}" "${CMAKE_CXX_FLAGS_DEBUG}" "${CMAKE_CXX_FLAGS}"
        "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS}" "${CMAKE_EXE_FLAGS}"
        "${LibSuffix}" "${SYSLIBS}"
        "${libdir}" "-lCore" "-lRint" "${incdir}" "" "" "${ROOT_ARCHITECTURE}" "" "${explicitlink}" )
endif()

configure_file(${CMAKE_SOURCE_DIR}/config/root-config.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/root-config @ONLY NEWLINE_STYLE UNIX)
configure_file(${CMAKE_SOURCE_DIR}/config/memprobe.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/memprobe @ONLY NEWLINE_STYLE UNIX)
configure_file(${CMAKE_SOURCE_DIR}/config/thisroot.sh ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/thisroot.sh @ONLY NEWLINE_STYLE UNIX)
configure_file(${CMAKE_SOURCE_DIR}/config/thisroot.csh ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/thisroot.csh @ONLY NEWLINE_STYLE UNIX)
configure_file(${CMAKE_SOURCE_DIR}/config/setxrd.csh ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/setxrd.csh COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/config/setxrd.sh ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/setxrd.sh COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/config/proofserv.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/proofserv @ONLY NEWLINE_STYLE UNIX)
configure_file(${CMAKE_SOURCE_DIR}/config/roots.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/roots @ONLY NEWLINE_STYLE UNIX)
configure_file(${CMAKE_SOURCE_DIR}/config/root-help.el.in root-help.el @ONLY NEWLINE_STYLE UNIX)
if (XROOTD_FOUND AND XROOTD_NOMAIN)
  configure_file(${CMAKE_SOURCE_DIR}/config/xproofd.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/xproofd @ONLY NEWLINE_STYLE UNIX)
endif()
if(WIN32)
  set(thisrootbat ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/thisroot.bat)
  configure_file(${CMAKE_SOURCE_DIR}/config/thisroot.bat ${thisrootbat} @ONLY)
endif()

install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/memprobe
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/thisroot.sh
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/thisroot.csh
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/setxrd.csh
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/setxrd.sh
              ${thisrootbat}
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/root-config
              ${CMAKE_SOURCE_DIR}/cmake/scripts/setenvwrap.csh
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/roots
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/proofserv
              PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                          GROUP_EXECUTE GROUP_READ
                          WORLD_EXECUTE WORLD_READ
              DESTINATION ${CMAKE_INSTALL_BINDIR})

if (XROOTD_FOUND AND XROOTD_NOMAIN)
   install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/xproofd
                 PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                             GROUP_EXECUTE GROUP_READ
                             WORLD_EXECUTE WORLD_READ
                 DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(FILES ${CMAKE_BINARY_DIR}/include/RConfigOptions.h
              ${CMAKE_BINARY_DIR}/include/compiledata.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${CMAKE_BINARY_DIR}/etc/root.mimes
              ${CMAKE_BINARY_DIR}/etc/system.rootrc
              DESTINATION ${CMAKE_INSTALL_SYSCONFDIR})

install(FILES ${CMAKE_BINARY_DIR}/root-help.el DESTINATION ${CMAKE_INSTALL_ELISPDIR})

if(NOT gnuinstall)
  install(FILES ${CMAKE_BINARY_DIR}/config/Makefile.comp
                ${CMAKE_BINARY_DIR}/config/Makefile.config
                DESTINATION config)
endif()


endfunction()
RootConfigure()
