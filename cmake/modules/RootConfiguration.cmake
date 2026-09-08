# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

INCLUDE (CheckCXXSourceCompiles)

#---Define a function not to pollute the top level namespace with unneeded variables-----------------------
function(RootConfigure)

#---Define all sort of variables to bridge between the old Module.mk and the new CMake equivalents-----------
foreach(v 1 ON YES TRUE Y on yes true y)
  set(value${v} yes)
endforeach()
foreach(v 0 OFF NO FALSE N IGNORE off no false n ignore)
  set(value${v} no)
endforeach()

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
if(IS_ABSOLUTE ${CMAKE_INSTALL_FONTDIR})
  set(ttffontdir ${CMAKE_INSTALL_FONTDIR})
else()
  set(ttffontdir ${prefix}/${CMAKE_INSTALL_FONTDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_JSROOTDIR})
  set(jsrootdir ${CMAKE_INSTALL_JSROOTDIR})
else()
  set(jsrootdir ${prefix}/${CMAKE_INSTALL_JSROOTDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_OPENUI5DIR})
  set(openui5dir ${CMAKE_INSTALL_OPENUI5DIR})
else()
  set(openui5dir ${prefix}/${CMAKE_INSTALL_OPENUI5DIR})
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
if(IS_ABSOLUTE ${CMAKE_INSTALL_DOCDIR})
  set(docdir ${CMAKE_INSTALL_DOCDIR})
else()
  set(docdir ${prefix}/${CMAKE_INSTALL_DOCDIR})
endif()
if(IS_ABSOLUTE ${CMAKE_INSTALL_TUTDIR})
  set(tutdir ${CMAKE_INSTALL_TUTDIR})
else()
  set(tutdir ${prefix}/${CMAKE_INSTALL_TUTDIR})
endif()

set(buildx11 ${value${x11}})
set(x11libdir -L${X11_LIBRARY_DIR})
set(xpmlibdir -L${X11_LIBRARY_DIR})
set(xpmlib ${X11_Xpm_LIB})

set(thread yes)
set(enable_thread yes)
set(threadflag ${CMAKE_THREAD_FLAG})
set(threadlibdir)
set(threadlib ${CMAKE_THREAD_LIBS_INIT})

set(builtinfreetype ${value${builtin_freetype}})
set(builtinpcre ${value${builtin_pcre}})

set(builtinzlib ${value${builtin_zlib}})
set(zliblibdir ${ZLIB_LIBRARY_DIR})
set(zliblib ${ZLIB_LIBRARY})
set(zlibincdir ${ZLIB_INCLUDE_DIR})

set(builtinunuran ${value${builtin_unuran}})
set(unuranlibdir ${UNURAN_LIBRARY_DIR})
set(unuranlib ${UNURAN_LIBRARY})
set(unuranincdir ${UNURAN_INCLUDE_DIR})

set(buildgl ${value${opengl}})
set(opengllibdir ${OPENGL_LIBRARY_DIR})
set(openglulib ${OPENGL_glu_LIBRARY})
set(opengllib ${OPENGL_gl_LIBRARY})
set(openglincdir ${OPENGL_INCLUDE_DIR})

set(builtingl2ps ${value${builtin_gl2ps}})
set(gl2pslibdir ${GL2PS_LIBRARY_DIR})
set(gl2pslib ${GL2PS_LIBRARY})
set(gl2psincdir ${GL2PS_INCLUDE_DIR})

set(buildsqlite ${value${sqlite}})
set(sqlitelibdir ${SQLITE_LIBRARY_DIR})
set(sqlitelib ${SQLITE_LIBRARY})
set(sqliteincdir ${SQLITE_INCLUDE_DIR})

set(builddavix ${value${davix}})
set(davixlibdir ${DAVIX_LIBRARY_DIR})
set(davixlib ${DAVIX_LIBRARY})
set(davixincdir ${DAVIX_INCLUDE_DIR})

set(buildnetxng ${value${xrootd}})

set(buildcurl ${value${curl}})
set(curllibdir ${CURL_LIBRARY_DIR})
set(curllib ${CURL_LIBRARY})
set(curlincdir ${CURL_INCLUDE_DIR})

set(builddcap ${value${dcap}})
set(dcaplibdir ${DCAP_LIBRARY_DIR})
set(dcaplib ${DCAP_LIBRARY})
set(dcapincdir ${DCAP_INCLUDE_DIR})

set(buildftgl ${value${builtin_ftgl}})
set(ftgllibdir ${FTGL_LIBRARY_DIR})
set(ftgllibs ${FTGL_LIBRARIES})
set(ftglincdir ${FTGL_INCLUDE_DIR})

set(buildarrow ${value${arrow}})
set(arrowlibdir ${ARROW_LIBRARY_DIR})
set(arrowlib ${ARROW_LIBRARY})
set(arrowincdir ${ARROW_INCLUDE_DIR})

set(buildasimage ${value${asimage}})
set(asextralib ${ASEXTRA_LIBRARIES})
set(asextralibdir)
set(asjpegincdir ${JPEG_INCLUDE_DIR})
set(aspngincdir ${PNG_INCLUDE_DIR})
set(astiffincdir ${TIFF_INCLUDE_DIR})
set(asgifincdir ${GIF_INCLUDE_DIR})
set(asimageincdir)
set(asimagelib)
set(asimagelibdir)

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

set(buildpython ${value${pyroot}})
set(pythonlibdir ${Python3_LIBRARY_DIR})
set(pythonlib ${Python3_LIBRARIES})
set(pythonincdir ${Python3_INCLUDE_DIRS})
set(pythonlibflags)

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

set(alloclib)
set(alloclibdir)

set(ssllib ${OPENSSL_LIBRARIES})
set(ssllibdir)
set(sslincdir ${OPENSSL_INCLUDE_DIR})
set(sslshared)

set(gsllibs ${GSL_LIBRARIES})
set(gsllibdir)
set(gslincdir ${GSL_INCLUDE_DIR})
set(gslflags)

set(shadowpw ${value${shadowpw}})
set(buildmathmore ${value${mathmore}})
set(buildroofit ${value${roofit}})
set(buildunuran ${value${unuran}})
set(buildgdml ${value${gdml}})
set(buildhttp ${value${http}})
if(fcgi AND http)
set(usefastcgi yes)
set(fastcgiincdir ${FASTCGI_INCLUDE_DIR})
else()
set(usefastcgi no)
set(fcgiincdir)
endif()

set(buildtmva ${value${tmva}})

set(cursesincdir ${CURSES_INCLUDE_DIR})
set(curseslibdir)
set(curseslib ${CURSES_LIBRARIES})
set(curseshdr ${CURSES_HEADER_FILE})
set(buildeditline ${value${editline}})
set(cppunit)


find_program(PERL_EXECUTABLE perl)
set(perl ${PERL_EXECUTABLE})

find_program(CHROME_EXECUTABLE NAMES chrome.exe chromium chromium-browser chrome chrome-browser google-chrome-stable Google\ Chrome
             HINTS /snap/bin
             PATH_SUFFIXES "Google/Chrome/Application")
if(CHROME_EXECUTABLE)
  if(WIN32)
    set(chromemajor 100)
    message(STATUS "Found Chrome browser executable ${CHROME_EXECUTABLE}, not testing the version")
  else()
    execute_process(COMMAND "${CHROME_EXECUTABLE}" --version
                    OUTPUT_VARIABLE CHROME_VERSION
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "[0-9]+" CHROME_MAJOR_VERSION "${CHROME_VERSION}")
    set(chromemajor ${CHROME_MAJOR_VERSION})
    message(STATUS "Found Chrome browser executable ${CHROME_EXECUTABLE} major version ${CHROME_MAJOR_VERSION}")
  endif()
  set(chromeexe ${CHROME_EXECUTABLE})
endif()

if(WIN32)
  find_program(EDGE_EXECUTABLE NAMES msedge.exe
               PATH_SUFFIXES "Microsoft/Edge/Application")
  if(EDGE_EXECUTABLE)
    message(STATUS "Found Edge browser executable ${EDGE_EXECUTABLE}")
    set(edgeexe ${EDGE_EXECUTABLE})
  endif()
endif()

find_program(FIREFOX_EXECUTABLE NAMES firefox firefox-bin firefox.exe
             HINTS /snap/bin
             PATH_SUFFIXES "Mozilla Firefox")
if(FIREFOX_EXECUTABLE)
  message(STATUS "Found Firefox browser executable ${FIREFOX_EXECUTABLE}")
  set(firefoxexe ${FIREFOX_EXECUTABLE})
endif()


#---RConfigure-------------------------------------------------------------------------------------------------
# set(setresuid undef)
CHECK_CXX_SOURCE_COMPILES("#include <unistd.h>
  int main() { uid_t r = 0, e = 0, s = 0; if (setresuid(r, e, s) != 0) { }; return 0;}" found_setresuid)

CHECK_CXX_SOURCE_COMPILES("
inline __attribute__((always_inline)) bool TestBit(unsigned long f) { return f != 0; };
int main() { return TestBit(0); }" found_attribute_always_inline)

CHECK_CXX_SOURCE_COMPILES("
inline __attribute__((noinline)) bool TestBit(unsigned long f) { return f != 0; };
int main() { return TestBit(0); }" has_found_attribute_noinline)

# The hardware interference size must be stable across all TUs in a ROOT build, so we need to save it in RConfigure.h
# Since it can vary for different compilers or tune settings, we cannot base the ABI on a value that might change,
# even be different between compiler and interpreter, or when ROOT is compiled on a different machine.
# For older CMake and when cross compiling, we simply fall back to 64
if(CMAKE_VERSION VERSION_GREATER 3.24 AND NOT CMAKE_CROSSCOMPILING)
set(test_interference_size "
#include <new>
#include <iostream>
int main() {
  std::cout << std::hardware_destructive_interference_size << std::endl;
  return 0;
}
")
try_run(HARDWARE_INTERF_RUN HARDWARE_INTERF_COMPILE
  SOURCE_FROM_VAR test_interference_size.cxx test_interference_size
  RUN_OUTPUT_VARIABLE hardwareinterferencesize)
endif()
if(NOT HARDWARE_INTERF_COMPILE OR NOT HARDWARE_INTERF_RUN EQUAL 0)
  message(STATUS "Could not detect hardware_interference_size in C++. Falling back to 64.")
  set(hardwareinterferencesize 64)
endif()

if(webgui)
   set(root_canvas_class "TWebCanvas")
   set(root_treeviewer_class "RTreeViewer")
   set(root_geompainter_type "web")
   set(root_jupyter_jsroot "on")
else()
   set(root_canvas_class "TRootCanvas")
   set(root_treeviewer_class "TTreeViewer")
   set(root_geompainter_type "root")
   set(root_jupyter_jsroot "off")
endif()

if(root7 AND webgui)
   set(root_browser_class "ROOT::RWebBrowserImp")
else()
   set(root_browser_class "TRootBrowser")
endif()

#---root-config----------------------------------------------------------------------------------------------
ROOT_GET_OPTIONS(features ENABLED)
set(features "cxx${CMAKE_CXX_STANDARD} ${features}")
set(configfeatures ${features})
set(configargs ${ROOT_CONFIGARGS})
set(configoptions ${ROOT_CONFIGARGS})
set(configstd ${CMAKE_CXX${CMAKE_CXX_STANDARD}_STANDARD_COMPILE_OPTION})
get_filename_component(altcc ${CMAKE_C_COMPILER} NAME)
get_filename_component(altcxx ${CMAKE_CXX_COMPILER} NAME)
get_filename_component(altf77 "${CMAKE_Fortran_COMPILER}" NAME)
get_filename_component(altld ${CMAKE_CXX_COMPILER} NAME)

set(pythonvers ${Python3_VERSION})
set(python${Python3_VERSION_MAJOR}vers ${Python3_VERSION})

#---RConfigure.h---------------------------------------------------------------------------------------------
if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
   execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dM -E /dev/null OUTPUT_VARIABLE __cplusplus_PPout)
else()
   try_compile(has__cplusplus "${CMAKE_BINARY_DIR}" SOURCES "${CMAKE_SOURCE_DIR}/config/__cplusplus.cxx"
            OUTPUT_VARIABLE __cplusplus_PPout)
endif()
string(REGEX MATCH "__cplusplus[=| ]([0-9]+)" __cplusplus "${__cplusplus_PPout}")
set(__cplusplus ${CMAKE_MATCH_1}L)

# To mark the build tree. Important for automatic resolution of relative paths,
# for example to the include directory. Use custom target to ensure re-creation
# when someone deletes the marker.
set(build_tree_marker "${localruntimedir}/root-build-tree-marker")
add_custom_command(
  OUTPUT "${build_tree_marker}"
  COMMAND ${CMAKE_COMMAND} -E touch "${build_tree_marker}"
  COMMENT "Ensuring that \"${build_tree_marker}\" exists"
)
add_custom_target(ensure_build_tree_marker ALL
  DEPENDS "${build_tree_marker}"
)

include(CheckSymbolExists)
include(CheckCXXCompilerFlag)
include(CheckSourceCompiles)

add_library(RConfigureDefs INTERFACE) # temporary target, do not link against it, just for bw-compatible header generation RConfigure.h
if (gnuinstall)
  target_compile_definitions(RConfigureDefs INTERFACE
    ROOTPREFIX="${prefix}"
    ROOTBINDIR="${bindir}"
    ROOTLIBDIR="${libdir}"
    ROOTETCDIR="${etcdir}"
    ROOTDATADIR="${datadir}"
    ROOTDOCDIR="${docdir}"
    ROOTMACRODIR="${macrodir}"
    ROOTTUTDIR="${tutdir}"
    ROOTSRCDIR="${srcdir}"
    ROOTICONPATH="${iconpath}"
    TTFFONTDIR="${ttffontdir}"
  )
endif()

target_compile_definitions(RConfigureDefs INTERFACE
  ROOT__ARCHITECTURE=${architecture}
  EXTRAICONPATH=$<IF:$<BOOL:${extraiconpath}>,\"${extraiconpath}\",\"\">
  ROOT__cplusplus=${__cplusplus}
  $<$<BOOL:${found_setresuid}>:R__HAS_SETRESUID>
  $<$<BOOL:${mathmore}>:R__HAS_MATHMORE>
  $<$<BOOL:${CMAKE_USE_PTHREADS_INIT}>:R__HAS_PTHREAD>
  $<$<BOOL:${x11}>:R__HAS_XFT>
  $<$<BOOL:${clad}>:R__HAS_CLAD>
  $<$<BOOL:${cocoa}>:R__HAS_COCOA>
  $<$<BOOL:${vdt}>:R__HAS_VDT>
  $<$<BOOL:${ROOT_HAVE_EXPERIMENTAL_SIMD}>:R__HAS_STD_EXPERIMENTAL_SIMD>
  $<$<BOOL:${ROOT_EXPERIMENTAL_SIMD_PIN_AVX_ABI}>:R__EXPERIMENTAL_SIMD_PIN_AVX_ABI>
  $<$<BOOL:${runtime_cxxmodules}>:R__USE_CXXMODULES>
  $<$<BOOL:${libcxx}>:R__USE_LIBCXX>
  $<$<BOOL:${found_attribute_always_inline}>:R__HAS_ATTRIBUTE_ALWAYS_INLINE>
  $<$<BOOL:${has_found_attribute_noinline}>:R__HAS_ATTRIBUTE_NOINLINE>
  $<$<BOOL:${imt}>:R__USE_IMT>
  $<$<BOOL:${memory_termination}>:R__COMPLETE_MEM_TERMINATION>
  $<$<BOOL:${cefweb}>:R__HAS_CEFWEB>
  $<$<BOOL:${qt6web}>:R__HAS_QT6WEB>
  $<$<BOOL:${davix}>:R__HAS_DAVIX>
  $<$<BOOL:${curl}>:R__HAS_CURL>
  $<$<BOOL:${dataframe}>:R__HAS_DATAFRAME>
  $<$<BOOL:${root7}>:R__HAS_ROOT7>
  $<$<BOOL:${dev}>:R__LESS_INCLUDES>
  R__HARDWARE_INTERFERENCE_SIZE=${hardwareinterferencesize}
  $<$<BOOL:${ZLIB_NG}>:R__HAS_ZLIB_NG>
  $<$<BOOL:${tmva-cpu}>:R__HAS_TMVACPU>
  $<$<BOOL:${tmva-gpu}>:R__HAS_TMVAGPU>
  $<$<BOOL:${tmva-cudnn}>:R__HAS_CUDNN>
  $<$<BOOL:${tmva-pymva}>:R__HAS_PYMVA>
  $<$<BOOL:${uring}>:R__HAS_URING>
  $<$<BOOL:${geom}>:R__HAS_GEOM>
)

file(GENERATE
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ginclude/RConfigure.h
    CONTENT
"#ifndef ROOT_RConfigure
#define ROOT_RConfigure

#define $<JOIN:$<LIST:TRANSFORM,$<TARGET_PROPERTY:RConfigureDefs,INTERFACE_COMPILE_DEFINITIONS>,REPLACE,=, >,\n#define >
#endif

#ifndef ROOT_RConfigure_w
#define ROOT_RConfigure_w
#if defined(__cplusplus) && (__cplusplus != ROOT__cplusplus)
# define R__STR(x) #x
# define R__XSTR(x) R__STR(x)
# pragma message(__FILE__ \": Warning: The C++ standard in this build (\" R__XSTR(__cplusplus) \") does not match ROOT configuration (\" R__XSTR(ROOT__cplusplus) \"); this might cause unexpected issues.\")
# if defined(_MSC_VER)
#  pragma message(__FILE__ \": Warning: And please make sure you are using the -Zc:__cplusplus compilation flag\")
# endif
# undef R__XSTR
# undef R__STR
#endif
#endif
"
    NEWLINE_STYLE UNIX
)
install(FILES ${CMAKE_BINARY_DIR}/ginclude/RConfigure.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# RVersion.hxx
add_library(RVersionDefs INTERFACE) # temporary target, do not link against it, just for bw-compatible header generation
target_compile_definitions(RVersionDefs INTERFACE
  ROOT_VERSION_MAJOR=${ROOT_MAJOR_VERSION}
  ROOT_VERSION_MINOR=${ROOT_MINOR_VERSION}
  ROOT_VERSION_PATCH=${ROOT_PATCH_VERSION}
  ROOT_RELEASE_DATE="${ROOT_RELEASE_DATE}"
  ROOT_RELEASE_TIME="00:00:00" # not updated anymore
)
file(GENERATE
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ginclude/ROOT/RVersion.hxx
    CONTENT
"#ifndef ROOT_RVERSION_HXX
#define ROOT_RVERSION_HXX

#define $<JOIN:$<LIST:TRANSFORM,$<TARGET_PROPERTY:RVersionDefs,INTERFACE_COMPILE_DEFINITIONS>,REPLACE,=, >,\n#define >
#endif // ROOT_RVERSION_HXX

#ifndef #ifndef ROOT_RVERSION_HXX_m
#define ROOT_RVERSION_HXX_m
/* Don't change the lines below. */

/*
 * These macros can be used in the following way:
 *
 *    #if ROOT_VERSION_CODE >= ROOT_VERSION(6,32,4)
 *       #include <newheader.h>
 *    #else
 *       #include <oldheader.h>
 *    #endif
 *
*/

#define ROOT_VERSION(a,b,c) (((a) << 16) + ((b) << 8) + (c))
#define ROOT_VERSION_CODE ROOT_VERSION(ROOT_VERSION_MAJOR, ROOT_VERSION_MINOR, ROOT_VERSION_PATCH)

#define R__VERS_QUOTE1_MAJOR(P) #P
#define R__VERS_QUOTE_MAJOR(P) R__VERS_QUOTE1_MAJOR(P)


#if ROOT_VERSION_MINOR < 10
#define R__VERS_QUOTE1_MINOR(P) \"0\" #P
#else
#define R__VERS_QUOTE1_MINOR(P) #P
#endif
#define R__VERS_QUOTE_MINOR(P) R__VERS_QUOTE1_MINOR(P)

#if ROOT_VERSION_PATCH < 10
#define R__VERS_QUOTE1_PATCH(P) \"0\" #P
#else
#define R__VERS_QUOTE1_PATCH(P) #P
#endif
#define R__VERS_QUOTE_PATCH(P) R__VERS_QUOTE1_PATCH(P)

#define ROOT_RELEASE R__VERS_QUOTE_MAJOR(ROOT_VERSION_MAJOR) \
   \".\" R__VERS_QUOTE_MINOR(ROOT_VERSION_MINOR) \
   \".\" R__VERS_QUOTE_PATCH(ROOT_VERSION_PATCH)
#endif // ROOT_RVERSION_HXX
"
    NEWLINE_STYLE UNIX
)

# ----- RConfig.hxx
check_cxx_compiler_flag("-fmodules" HAS_MODULES)

# Machines
# TODO all this part can be easily simplified by using native CMake system name checks
check_source_compiles(CXX "
#if !defined(__hpux)
#error \"This is not HP-UX\"
#endif
int main() { return 0; }
" HAS_HPUX)
check_source_compiles(CXX "
#if !defined(__LP64__) || !defined(__hpux)
#error This is not HP-UX LP64
#endif
int main() { return 0; }
" HAS_LP64)
check_source_compiles(CXX "
#if !defined(__hpux)
#error \"This is not HP-UX\"
#endif
int main() { return 0; }
" HAS_HPUX)
check_source_compiles(CXX "
#if defined(__linux) || defined(__linux__) || (defined(__CYGWIN__) && defined(__GNUC__))
#   ifdef linux
#error \"This does not need Linux\"
#   endif
#else
#error \"This is not Linux\"
#endif
int main() { return 0; }
" NEEDS_LINUX)
check_source_compiles(CXX "
#if !defined(__linux) && defined(__linux__) && !defined(linux) && !(defined(__CYGWIN__) && defined(__GNUC__))
#error \"This is not Linux\"
#endif
int main() { return 0; }
" IS_LINUX)
check_source_compiles(CXX "
#if !defined(__CYGWIN__) || !defined(__GNUC__)
#error \"This is not wingcc\"
#endif
int main() { return 0; }
" HAS_WINGCC)
check_source_compiles(CXX "
#if defined(__sun) && !(defined(linux) || defined(__FCC_VERSION)) && defined(__SVR4)
#else
#error \"This is not Solaris\"
#endif
int main() { return 0; }
" HAS_SOLARIS)
check_source_compiles(CXX "
#if defined(__sun) && !(defined(linux) || defined(__FCC_VERSION)) && !defined(__SVR4)
#else
#error \"This is not Sun\"
#endif
int main() { return 0; }
" HAS_SUN)
check_symbol_exists(siglongjmp "setjmp.h" HAVE_SIGLONGJMP)
check_symbol_exists(lstat64 "sys/stat.h;sys/types.h" HAVE_LSTAT64)
check_source_compiles(CXX "
#if (!defined(__linux) && !defined(__linux__) && !defined(linux)) || defined(_LARGEFILE64_SOURCE)
#error \"This does not need _LARGEFILE64_SOURCE\"
#endif
int main() { return 0; }
" NEEDS_LARGEFILE64)
check_symbol_exists(strlcpy "string.h" HAS_STRLCPY)
check_symbol_exists(strcasecmp "strings.h;string.h" HAS_STRCASECMP)
check_source_compiles(CXX "
#if (!defined(__linux) && !defined(__linux__) && !defined(linux)) || !defined(i386) || defined(__i486__)
#error \"This does not need i486\"
#endif
int main() { return 0; }
" NEEDS_I486)
check_source_compiles(CXX "
#if defined(_INCLUDE_LONGLONG) || !defined(__HP_aCC)
#error \"This does not need longlong\"
#endif
int main() { return 0; }
" NEEDS_INC_LONGLONG)
check_source_compiles(CXX "
#if defined(WIN32) || !defined(_WIN32)
#error \"This does not need WIN32\"
#endif
int main() { return 0; }
" NEEDS_WIN32)
check_source_compiles(CXX "
#if defined(WIN64) || !defined(_WIN64)
#error \"This does not need WIN64\"
#endif
int main() { return 0; }
" NEEDS_WIN64)

add_library(RConfigDefs INTERFACE) # temporary target, do not link against it, just for bw-compatible header generation
target_compile_definitions(RConfigDefs INTERFACE
  $<$<BOOL:${HAS_MODULES}>:R__CXXMODULES>
  R__USE_SHADOW_CLASS
  R__ANSISTREAM
  R_SSTREAM
  R__NULLPTR
  # In comments below potential simplifications using native CMake
  $<$<BOOL:${HAS_HPUX}>:R__HPUX>
  $<$<BOOL:${HAS_WINGCC}>:R__WINGCC> # $<$<AND:$<PLATFORM_ID:Cygwin>,$<CXX_COMPILER_ID:GNU>>:linux;R__WINGCC>
  $<$<BOOL:${HAS_SOLARIS}>:R__SOLARIS> # CMAKE_SYSTEM_NAME STREQUAL "SunOS"
  $<$<BOOL:${HAS_SUN}>:R__SUN> # CMAKE_SYSTEM_NAME STREQUAL "SunOS"
  $<$<BOOL:${UNIX}>:R__UNIX>
  $<$<BOOL:${IS_LINUX}>:R__LINUX>
  $<$<BOOL:${NEEDS_LINUX}>:linux>
  $<$<BOOL:${HAVE_LSTAT64}>:R__SEEK64>
  $<$<BOOL:${HAVE_SIGLONGJMP}>:NEED_SIGJMP>
  $<$<STREQUAL:${CMAKE_CXX_BYTE_ORDER},LITTLE_ENDIAN>:R__BYTESWAP>
  $<$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>:R__B64>
  $<$<BOOL:${NEEDS_LARGEFILE64}>:_LARGEFILE64_SOURCE>
  $<$<BOOL:${HAS_STRLCPY}>:HAS_STRLCPY=1> # TODO move this just to Clib and remove from global defs
  $<$<BOOL:${NEEDS_I486}>:__i486__>
  $<$<AND:$<STREQUAL:${CMAKE_SYSTEM_NAME},GNU/Hurd>,$<EQUAL:${CMAKE_SIZEOF_VOID_P},4>>:R__HURD;f2cFortran>
  $<$<PLATFORM_ID:FreeBSD>:R__FBSD>
  $<$<PLATFORM_ID:OpenBSD>:R__OBSD>
  $<$<PLATFORM_ID:Apple>:R__MACOSX>
  $<$<STREQUAL:${CMAKE_SYSTEM_NAME},HI-UX>:R__HIUX>
  $<$<NOT:$<BOOL:${HAS_STRCASECMP}>>:NEED_STRCASECMP> # could be moved to a private implementation detail into TString.cxx
  $<$<AND:$<STREQUAL:${CMAKE_SYSTEM_NAME},LynxOS>,$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},powerpc>>:R__LYNXOS>
  # $<$<C_COMPILER_ID:GNU>:R__HIDDEN=__attribute__((__visibility__(\"hidden\")))>> # Not used
  # $<$<OR:$<C_COMPILER_ID:Intel>,$<C_COMPILER_ID:IntelLLVM>>:R__INTEL_COMPILER> # Not used
  $<$<CXX_COMPILER_ID:HP>:R__ACC;R__TMPLTSTREAM>
  $<$<BOOL:${NEEDS_INC_LONGLONG}>:_INCLUDE_LONGLONG>
  $<$<PLATFORM_ID:Windows>:R__WIN32> # R__ACCESS_IN_SYMBOL not used
  $<$<BOOL:${NEEDS_WIN32}>:WIN32>
  $<$<AND:$<PLATFORM_ID:Windows>,$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>>:R__WIN64> # R__x86_64__ not used
  $<$<BOOL:${NEEDS_WIN64}>:WIN64>
  $<$<CXX_COMPILER_ID:Symantec>:SC;R__SC>
  $<$<AND:$<CXX_COMPILER_ID:Symantec>,$<NOT:$<PLATFORM_ID:Windows>>>:MSDOS> # TODO move to Zlib private impreventation detail
  $<$<CXX_COMPILER_ID:MSVC>:R__VISUAL_CPLUSPLUS>
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,13.1>>:R__NO_CLASS_TEMPLATE_SPECIALIZATION>
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<VERSION_LESS_EQUAL:$<CXX_COMPILER_VERSION>,18.0>>:R__NO_ATOMIC_FUNCTION_POINTER>
  $<$<BOOL:${ENABLE_BASKET_ALLOC_TIME_TRACKING}>:R__TRACK_BASKET_ALLOC_TIME=1>
)
file(GENERATE
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ginclude/ROOT/RConfig.hxx
    CONTENT
"
#ifndef ROOT_RConfig_h
#define ROOT_RConfig_h
#include <ROOT/RVersion.hxx>
#include <RConfigure.h>
#endif

#ifndef ROOT_RConfig
#define ROOT_RConfig

#define $<JOIN:$<LIST:TRANSFORM,$<TARGET_PROPERTY:RConfigDefs,INTERFACE_COMPILE_DEFINITIONS>,REPLACE,=, >,\n#define >
#endif

#ifndef ROOT_RConfig_m
#define ROOT_RConfig_m
#   define _NAME1_(name) name
#   define _NAME2_(name1,name2) name1##name2
#   define _NAME3_(name1,name2,name3) name1##name2##name3

    /* stringizing */
#   define _QUOTE_(name) #name
#define _R_QUOTEVAL_(string) _QUOTE_(string)
/* produce an identifier that is almost unique inside a file */
#   define _R__JOIN_(X,Y) _NAME2_(X,Y)
#   define _R__JOIN3_(F,X,Y) _NAME3_(F,X,Y)
#   define _R__UNIQUE_DICT_(X) _R__JOIN3_(R__DICTIONARY_FILENAME,X,__LINE__)
#   define _R__UNIQUE_(X) _R__JOIN_(X,__LINE__)
#endif

#ifndef ROOT_RConfig_w
#define ROOT_RConfig_w

#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
# if (__GNUC__ == 5 && (__GNUC_MINOR__ == 1 || __GNUC_MINOR__ == 2)) || defined(R__NO_DEPRECATION)
/* GCC 5.1, 5.2: false positives due to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=15269
   or deprecation turned off */
#   define _R__DEPRECATED_LATER(REASON)
# else
#   define _R__DEPRECATED_LATER(REASON) __attribute__((deprecated(REASON)))
# endif
#elif defined(_MSC_VER) && !defined(R__NO_DEPRECATION)
#   define _R__DEPRECATED_LATER(REASON) __pragma(deprecated(REASON))
#else
/* Deprecation not supported for this compiler. */
#   define _R__DEPRECATED_LATER(REASON)
#endif

#ifdef R__WIN32
#define _R_DEPRECATED_REMOVE_NOW(REASON)
#else
#define _R_DEPRECATED_REMOVE_NOW(REASON) __attribute__((REMOVE_THIS_NOW))
#endif

/* USE AS `R__DEPRECATED(6,42, \"Not threadsafe; use TFoo::Bar().\")`
   To be removed by 6.42 */
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 41, 2)
#define _R__DEPRECATED_642(REASON) _R__DEPRECATED_LATER(REASON)
#else
#define _R__DEPRECATED_642(REASON) _R_DEPRECATED_REMOVE_NOW(REASON)
#endif

#if ROOT_VERSION_CODE <= ROOT_VERSION(6, 43, 0)
#define _R__DEPRECATED_644(REASON) _R__DEPRECATED_LATER(REASON)
#else
#define _R__DEPRECATED_644(REASON) _R_DEPRECATED_REMOVE_NOW(REASON)
#endif

#if ROOT_VERSION_CODE <= ROOT_VERSION(6, 45, 0)
#define _R__DEPRECATED_646(REASON) _R__DEPRECATED_LATER(REASON)
#else
#define _R__DEPRECATED_646(REASON) _R_DEPRECATED_REMOVE_NOW(REASON)
#endif

/* USE AS `R__DEPRECATED(7,00, \"Not threadsafe; use TFoo::Bar().\")`
   To be removed by 7.00 */
#if ROOT_VERSION_CODE < ROOT_VERSION(6,99,0)
# define _R__DEPRECATED_700(REASON) _R__DEPRECATED_LATER(REASON)
#else
# define _R__DEPRECATED_700(REASON) _R_DEPRECATED_REMOVE_NOW(REASON)
#endif

/* Spell as R__DEPRECATED(6,04, \"Not threadsafe; use TFoo::Bar().\") */
#define R__DEPRECATED(MAJOR, MINOR, REASON) \
  _R__JOIN3_(_R__DEPRECATED_,MAJOR,MINOR)(\"will be removed in ROOT v\" #MAJOR \".\" #MINOR \": \" REASON)

/* Mechanisms to advise users to avoid legacy functions and classes that will not be removed */
#if defined R__SUGGEST_NEW_INTERFACE
#  define R__SUGGEST_ALTERNATIVE(ALTERNATIVE) \
      _R__DEPRECATED_LATER(\"There is a superior alternative: \" ALTERNATIVE)
#else
#  define R__SUGGEST_ALTERNATIVE(ALTERNATIVE)
#endif

#define R__ALWAYS_SUGGEST_ALTERNATIVE(ALTERNATIVE) \
    _R__DEPRECATED_LATER(\"There is a superior alternative: \" ALTERNATIVE)

/*---- misc ------------------------------------------------------------------*/

#ifdef R__GNU
#   define SafeDelete(p) { if (p) { delete p; p = nullptr; } }
#else
#   define SafeDelete(p) { delete p; p = nullptr; }
#endif

#ifdef __FAST_MATH__
#define R__FAST_MATH
#endif

#if (__GNUC__ >= 7)
#define R__DO_PRAGMA(x) _Pragma (#x)
# define R__INTENTIONALLY_UNINIT_BEGIN \
  R__DO_PRAGMA(GCC diagnostic push) \
  R__DO_PRAGMA(GCC diagnostic ignored \"-Wmaybe-uninitialized\") \
  R__DO_PRAGMA(GCC diagnostic ignored \"-Wuninitialized\")
# define R__INTENTIONALLY_UNINIT_END \
  R__DO_PRAGMA(GCC diagnostic pop)
#else
# define R__INTENTIONALLY_UNINIT_BEGIN
# define R__INTENTIONALLY_UNINIT_END

#endif

#ifdef R__HAS_ATTRIBUTE_ALWAYS_INLINE
#define R__ALWAYS_INLINE inline __attribute__((always_inline))
#else
#if defined(_MSC_VER)
#define R__ALWAYS_INLINE __forceinline
#else
#define R__ALWAYS_INLINE inline
#endif
#endif

// See also https://nemequ.github.io/hedley/api-reference.html#HEDLEY_NEVER_INLINE
// for other platforms.
#ifdef R__HAS_ATTRIBUTE_NOINLINE
#define R__NEVER_INLINE inline __attribute__((noinline))
#else
#if defined(_MSC_VER)
#define R__NEVER_INLINE inline  __declspec(noinline)
#else
#define R__NEVER_INLINE inline
#endif
#endif

/*---- unlikely / likely expressions -----------------------------------------*/
// These are meant to use in cases like:
//   if (R__unlikely(expression)) { ... }
// in performance-critical sections.  R__unlikely / R__likely provide hints to
// the compiler code generation to heavily optimize one side of a conditional,
// causing the other branch to have a heavy performance cost.
//
// It is best to use this for conditionals that test for rare error cases or
// backward compatibility code.

#if (__GNUC__ >= 3) || defined(__INTEL_COMPILER)
#if !defined(R__unlikely)
  #define R__unlikely(expr) __builtin_expect(!!(expr), 0)
#endif
#if !defined(R__likely)
  #define R__likely(expr) __builtin_expect(!!(expr), 1)
#endif
#else
  #define R__unlikely(expr) expr
  #define R__likely(expr) expr
#endif

#ifdef __HP_aCC
#   if __HP_aCC <= 015000
#error \"ROOT requires proper support for C++17 or higher\"
#   endif
#endif

#if defined(_MSC_VER)
# if (_MSC_VER < 1910)
#  error \"ROOT requires Visual Studio 2017 or higher.\"
#else
#if defined(__cplusplus) && (__cplusplus < 201703L)
#error \"ROOT requires support for C++17 or higher.\"
#  if defined(__GNUC__) || defined(__clang__)
#error \"Pass `-std=c++17` as compiler argument.\"
#  endif
# endif
#endif

#endif
"
    NEWLINE_STYLE UNIX
)

# Public target interface against which to link
add_library(ROOTdefs INTERFACE)
target_compile_definitions(ROOTdefs INTERFACE
ROOT_RConfigure # so that including the mirror header RConfigure.h is innocuous if linking against this target
ROOT_RVERSION_HXX
)
target_link_libraries(ROOTdefs INTERFACE RConfigureDefs RVersionDefs RConfigDefs)


#---Configure and install various files----------------------------------------------------------------------
execute_Process(COMMAND hostname OUTPUT_VARIABLE BuildNodeInfo OUTPUT_STRIP_TRAILING_WHITESPACE )

configure_file(${CMAKE_SOURCE_DIR}/config/rootrc.in ${CMAKE_BINARY_DIR}/etc/system.rootrc @ONLY NEWLINE_STYLE UNIX)

# file used in TROOT.cxx, not need in include/ dir and not need to install
configure_file(${CMAKE_SOURCE_DIR}/config/RConfigOptions.in ginclude/RConfigOptions.h NEWLINE_STYLE UNIX)

configure_file(${CMAKE_SOURCE_DIR}/config/Makefile-comp.in config/Makefile.comp NEWLINE_STYLE UNIX) # Will be removed in future release
configure_file(${CMAKE_SOURCE_DIR}/config/Makefile.in config/Makefile.config NEWLINE_STYLE UNIX) # Will be removed in future release
configure_file(${CMAKE_SOURCE_DIR}/config/mimes.unix.in ${CMAKE_BINARY_DIR}/etc/root.mimes NEWLINE_STYLE UNIX)
# We need to have class.rules during configuration time to avoid silent error during generation of dictionary:
# Error in <TClass::ReadRules()>: Cannot find rules
configure_file(${CMAKE_SOURCE_DIR}/etc/class.rules ${CMAKE_BINARY_DIR}/etc/class.rules COPYONLY)

#---Generate the ROOTConfig files to be used by CMake projects-----------------------------------------------
ROOT_GET_OPTIONS(ROOT_ALL_OPTIONS)
ROOT_GET_OPTIONS(ROOT_ENABLED_OPTIONS ENABLED)
configure_file(${CMAKE_SOURCE_DIR}/cmake/scripts/ROOTConfig-version.cmake.in
               ${CMAKE_BINARY_DIR}/ROOTConfig-version.cmake @ONLY NEWLINE_STYLE UNIX)

#---Compiler flags (because user apps are a bit dependent on them...)----------------------------------------
string(REGEX REPLACE "(^|[ ]*)-W[^ ]*" "" __cxxflags "${CMAKE_CXX_FLAGS}")
string(REGEX REPLACE "(^|[ ]*)-W[^ ]*" "" __cflags "${CMAKE_C_FLAGS}")

if(MSVC)
  string(REPLACE "-I${CMAKE_SOURCE_DIR}/cmake/win" "" __cxxflags "${__cxxflags}")
  string(REPLACE "-I${CMAKE_SOURCE_DIR}/cmake/win" "" __cflags "${__cflags}")
endif()

string(REGEX REPLACE "(^|[ ]*)-W[^ ]*" "" __fflags "${CMAKE_Fortran_FLAGS}")
string(REGEX MATCHALL "(-Wp,)?-(D|U)[^ ]*" __defs "${CMAKE_CXX_FLAGS}")
set(ROOT_COMPILER_FLAG_HINTS "#
set(ROOT_DEFINITIONS \"${__defs}\")
set(ROOT_CXX_STANDARD ${CMAKE_CXX_STANDARD})
set(ROOT_CXX_FLAGS \"${__cxxflags}\")
set(ROOT_C_FLAGS \"${__cflags}\")
set(ROOT_fortran_FLAGS \"${__fflags}\")
set(ROOT_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS}\")")
set(ROOT_BINDIR ${CMAKE_BINARY_DIR}/bin CACHE INTERNAL "")

#---To be used from the binary tree--------------------------------------------------------------------------
set(ROOT_INCLUDE_DIR_SETUP "
# ROOT configured for use from the build tree - absolute paths are used.
set(ROOT_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/include)
")
set(ROOT_LIBRARY_DIR_SETUP "
# ROOT configured for use from the build tree - absolute paths are used.
set(ROOT_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib)
")
set(ROOT_BINDIR_SETUP "
# ROOT configured for use from the build tree - absolute paths are used.
set(ROOT_BINDIR ${CMAKE_BINARY_DIR}/bin)
")
# Deprecated value ROOT_BINARY_DIR
set(ROOT_BINARY_DIR_SETUP "
# Deprecated value, please don't use it and use ROOT_BINDIR instead.
set(ROOT_BINARY_DIR ${ROOT_BINDIR})
")
set(ROOT_CMAKE_DIR_SETUP "
# ROOT configured for use from the build tree - absolute paths are used.
set(ROOT_CMAKE_DIR ${CMAKE_SOURCE_DIR}/cmake)
")

get_property(exported_targets GLOBAL PROPERTY ROOT_EXPORTED_TARGETS)
export(TARGETS ${exported_targets} NAMESPACE ROOT:: FILE ${PROJECT_BINARY_DIR}/ROOTConfig-targets.cmake)
configure_file(${CMAKE_SOURCE_DIR}/cmake/scripts/ROOTConfig.cmake.in
               ${CMAKE_BINARY_DIR}/ROOTConfig.cmake @ONLY NEWLINE_STYLE UNIX)
configure_file(${CMAKE_SOURCE_DIR}/cmake/scripts/RootUseFile.cmake.in
               ${CMAKE_BINARY_DIR}/ROOTUseFile.cmake @ONLY NEWLINE_STYLE UNIX)

#---To be used from the install tree--------------------------------------------------------------------------
# Need to calculate actual relative paths from CMAKEDIR to other locations
cmake_path(RELATIVE_PATH CMAKE_INSTALL_FULL_INCLUDEDIR BASE_DIRECTORY "${CMAKE_INSTALL_FULL_CMAKEDIR}" OUTPUT_VARIABLE ROOT_CMAKE_TO_INCLUDE_DIR)
cmake_path(RELATIVE_PATH CMAKE_INSTALL_FULL_LIBDIR BASE_DIRECTORY "${CMAKE_INSTALL_FULL_CMAKEDIR}" OUTPUT_VARIABLE ROOT_CMAKE_TO_LIB_DIR)
cmake_path(RELATIVE_PATH CMAKE_INSTALL_FULL_BINDIR BASE_DIRECTORY "${CMAKE_INSTALL_FULL_CMAKEDIR}" OUTPUT_VARIABLE ROOT_CMAKE_TO_BIN_DIR)

# '_' prefixed variables are used to construct the paths,
# while the normal variants evaluate to full paths at runtime
set(ROOT_INCLUDE_DIR_SETUP "
# ROOT configured for the install with relative paths, so use these
get_filename_component(_ROOT_INCLUDE_DIRS \"\${_thisdir}/${ROOT_CMAKE_TO_INCLUDE_DIR}\" REALPATH)
# resolve relative paths to absolute system paths
get_filename_component(ROOT_INCLUDE_DIRS \"\${_ROOT_INCLUDE_DIRS}\" REALPATH)
")
set(ROOT_LIBRARY_DIR_SETUP "
# ROOT configured for the install with relative paths, so use these
get_filename_component(_ROOT_LIBRARY_DIR \"\${_thisdir}/${ROOT_CMAKE_TO_LIB_DIR}\" REALPATH)
# resolve relative paths to absolute system paths
get_filename_component(ROOT_LIBRARY_DIR \"\${_ROOT_LIBRARY_DIR}\" REALPATH)
")
set(ROOT_BINDIR_SETUP "
# ROOT configured for the install with relative paths, so use these
get_filename_component(_ROOT_BINDIR \"\${_thisdir}/${ROOT_CMAKE_TO_BIN_DIR}\" REALPATH)
# resolve relative paths to absolute system paths
get_filename_component(ROOT_BINDIR \"\${_ROOT_BINDIR}\" REALPATH)
")
# Deprecated value ROOT_BINARY_DIR
set(ROOT_BINARY_DIR_SETUP "
# Deprecated value, please don't use it and use ROOT_BINDIR instead.
get_filename_component(ROOT_BINARY_DIR \"\${ROOT_BINDIR}\" REALPATH)
")
set(ROOT_CMAKE_DIR_SETUP "
## ROOT configured for the install with relative paths, so use these
get_filename_component(ROOT_CMAKE_DIR \"\${_thisdir}\" REALPATH)
")

# used by ROOTConfig.cmake from the build directory
configure_file(${CMAKE_SOURCE_DIR}/cmake/modules/RootMacros.cmake
               ${CMAKE_BINARY_DIR}/RootMacros.cmake COPYONLY)

# used by roottest to run tests against ROOT build
configure_file(${CMAKE_SOURCE_DIR}/cmake/modules/RootTestDriver.cmake
               ${CMAKE_BINARY_DIR}/RootTestDriver.cmake COPYONLY)

configure_file(${CMAKE_SOURCE_DIR}/cmake/scripts/ROOTConfig.cmake.in
               ${CMAKE_BINARY_DIR}/installtree/ROOTConfig.cmake @ONLY NEWLINE_STYLE UNIX)
configure_file(${CMAKE_SOURCE_DIR}/cmake/scripts/RootUseFile.cmake.in
               ${CMAKE_BINARY_DIR}/installtree/ROOTUseFile.cmake @ONLY NEWLINE_STYLE UNIX)
install(FILES ${CMAKE_BINARY_DIR}/ROOTConfig-version.cmake
              ${CMAKE_BINARY_DIR}/installtree/ROOTUseFile.cmake
              ${CMAKE_BINARY_DIR}/installtree/ROOTConfig.cmake DESTINATION ${CMAKE_INSTALL_CMAKEDIR})
install(EXPORT ${CMAKE_PROJECT_NAME}Exports NAMESPACE ROOT:: FILE ROOTConfig-targets.cmake DESTINATION ${CMAKE_INSTALL_CMAKEDIR})


#---Especial definitions for root-config et al.--------------------------------------------------------------
if(prefix STREQUAL "$(ROOTSYS)")
  foreach(d prefix bindir libdir incdir etcdir tutdir mandir)
    string(REPLACE "$(ROOTSYS)" "$ROOTSYS"  ${d} ${${d}})
  endforeach()
endif()


#---compiledata.h--------------------------------------------------------------------------------------------

# ROOTBUILD definition (it is defined in compiledata.h and used by ACLIC
# to decide whether (by default) to optimize or not optimize the user scripts.)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(ROOTBUILD "debug")
endif()

if(WIN32)
  # We cannot use the compiledata.sh script for windows
  configure_file(${CMAKE_SOURCE_DIR}/cmake/scripts/compiledata.win32.in ${CMAKE_BINARY_DIR}/ginclude/compiledata.h NEWLINE_STYLE UNIX)
else()
  # Needed by ACLIC, while in ROOT we are using everywhere C++ standard via CMake features that are requested to build target
  set(CMAKE_CXX_ACLIC_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX${CMAKE_CXX_STANDARD}_STANDARD_COMPILE_OPTION}")
  if(asan)
    set(CMAKE_CXX_ACLIC_FLAGS "${CMAKE_CXX_ACLIC_FLAGS} ${ASAN_EXTRA_CXX_FLAGS}")
  endif()
  if(ROOT_COMPILEDATA_IGNORE_BUILD_NODE_CHANGES)
    # Only set the compiledata parameter if the CMake variable is 'true'
    set(local_ROOT_COMPILEDATA_IGNORE_BUILD_NODE_CHANGES ${ROOT_COMPILEDATA_IGNORE_BUILD_NODE_CHANGES})
  endif()
  execute_process(COMMAND ${CMAKE_SOURCE_DIR}/cmake/unix/compiledata.sh
    ${CMAKE_BINARY_DIR}/ginclude/compiledata.h "${CMAKE_CXX_COMPILER}"
        "${CMAKE_CXX_FLAGS_RELEASE}" "${CMAKE_CXX_FLAGS_DEBUG}" "${CMAKE_CXX_ACLIC_FLAGS}"
        "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS}" "${CMAKE_EXE_LINKER_FLAGS}" "so"
        "${libdir}" "-lCore" "-lRint" "" "" "${ROOT_ARCHITECTURE}" "${ROOTBUILD}"
        "${local_ROOT_COMPILEDATA_IGNORE_BUILD_NODE_CHANGES}")
endif()

#---Get the values of ROOT_ALL_OPTIONS and CMAKE_CXX_FLAGS provided by the user in the command line
set(all_features ${ROOT_ALL_OPTIONS})
set(usercflags ${CMAKE_CXX_FLAGS-CACHED})
file(REMOVE ${CMAKE_BINARY_DIR}/installtree/root-config)
configure_file(${CMAKE_SOURCE_DIR}/config/root-config.in ${CMAKE_BINARY_DIR}/installtree/root-config @ONLY NEWLINE_STYLE UNIX)
if(thisroot_scripts)
  configure_file(${CMAKE_SOURCE_DIR}/config/thisroot.sh ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/thisroot.sh @ONLY NEWLINE_STYLE UNIX)
  configure_file(${CMAKE_SOURCE_DIR}/config/thisroot.csh ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/thisroot.csh @ONLY NEWLINE_STYLE UNIX)
  configure_file(${CMAKE_SOURCE_DIR}/config/thisroot.fish ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/thisroot.fish @ONLY NEWLINE_STYLE UNIX)
  list(APPEND list_of_thisroot_scripts thisroot.sh thisroot.csh thisroot.fish setxrd.csh)
endif()
configure_file(${CMAKE_SOURCE_DIR}/config/setxrd.csh ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/setxrd.csh COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/config/setxrd.sh ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/setxrd.sh COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/config/roots.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/roots @ONLY NEWLINE_STYLE UNIX)
configure_file(${CMAKE_SOURCE_DIR}/config/rootssh ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rootssh @ONLY NEWLINE_STYLE UNIX)
if(WIN32)
  if(thisroot_scripts)
    configure_file(${CMAKE_SOURCE_DIR}/config/thisroot.bat ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/thisroot.bat @ONLY)
    configure_file(${CMAKE_SOURCE_DIR}/config/thisroot.ps1 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/thisroot.ps1 @ONLY)
    list(APPEND list_of_thisroot_scripts thisroot.bat thisroot.ps1)
  endif()
  configure_file(${CMAKE_SOURCE_DIR}/config/root.rc.in ${CMAKE_BINARY_DIR}/etc/root.rc @ONLY)
  configure_file(${CMAKE_SOURCE_DIR}/config/root-manifest.xml.in ${CMAKE_BINARY_DIR}/etc/root-manifest.xml @ONLY)
  install(FILES ${CMAKE_SOURCE_DIR}/cmake/win/w32pragma.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers)
  install(FILES ${CMAKE_SOURCE_DIR}/cmake/win/sehmap.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT headers)
endif()

#--Local root-configure
set(prefix $ROOTSYS)
set(bindir $ROOTSYS/bin)
set(libdir $ROOTSYS/lib)
set(incdir $ROOTSYS/include)
set(etcdir $ROOTSYS/etc)
set(tutdir $ROOTSYS/tutorials)
set(mandir $ROOTSYS/man)
configure_file(${CMAKE_SOURCE_DIR}/config/root-config.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/root-config @ONLY NEWLINE_STYLE UNIX)
if(MSVC)
  configure_file(${CMAKE_SOURCE_DIR}/config/root-config.bat.in ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/root-config.bat @ONLY)
  install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/root-config.bat
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
              GROUP_EXECUTE GROUP_READ
              WORLD_EXECUTE WORLD_READ
  DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(DEFINED list_of_thisroot_scripts)
  # Prepend runtime output directory to list of setup scripts
  set(final_list_of_thisroot_scripts "")
  foreach(script IN LISTS list_of_thisroot_scripts)
      list(APPEND final_list_of_thisroot_scripts "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${script}")
  endforeach()

  install(FILES ${final_list_of_thisroot_scripts}
                PERMISSIONS OWNER_WRITE OWNER_READ
                            GROUP_READ
                            WORLD_READ
                DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(FILES ${CMAKE_BINARY_DIR}/installtree/root-config
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/roots
              ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/rootssh
              PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                          GROUP_EXECUTE GROUP_READ
                          WORLD_EXECUTE WORLD_READ
              DESTINATION ${CMAKE_INSTALL_BINDIR})

install(FILES ${CMAKE_BINARY_DIR}/ginclude/RConfigOptions.h
              ${CMAKE_BINARY_DIR}/ginclude/compiledata.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${CMAKE_BINARY_DIR}/etc/root.mimes
              ${CMAKE_BINARY_DIR}/etc/system.rootrc
              DESTINATION ${CMAKE_INSTALL_SYSCONFDIR})

endfunction()
RootConfigure()
