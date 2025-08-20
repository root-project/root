#ifndef ROOT_RConfigure
#define ROOT_RConfigure

/* Configurations file for linuxx8664gcc */

/* #undef R__HAVE_CONFIG */

#ifdef R__HAVE_CONFIG
#define ROOTPREFIX    "$(ROOTSYS)"
#define ROOTBINDIR    "$(ROOTSYS)/bin"
#define ROOTLIBDIR    "$(ROOTSYS)/lib"
#define ROOTINCDIR    "$(ROOTSYS)/include"
#define ROOTETCDIR    "$(ROOTSYS)/etc"
#define ROOTDATADIR   "$(ROOTSYS)/."
#define ROOTDOCDIR    "$(ROOTSYS)/."
#define ROOTMACRODIR  "$(ROOTSYS)/macros"
#define ROOTTUTDIR    "$(ROOTSYS)/tutorials"
#define ROOTSRCDIR    "$(ROOTSYS)/src"
#define ROOTICONPATH  "$(ROOTSYS)/icons"
#define TTFFONTDIR    "$(ROOTSYS)/fonts"
#endif

#define EXTRAICONPATH ""

#define ROOT__cplusplus 201703L
#if defined(__cplusplus) && (__cplusplus != ROOT__cplusplus)
# if defined(_MSC_VER)
#  pragma message("The C++ standard in this build does not match ROOT configuration (201703L); this might cause unexpected issues. And please make sure you are using the -Zc:__cplusplus compilation flag")
# else
#  warning "The C++ standard in this build does not match ROOT configuration (201703L); this might cause unexpected issues"
# endif
#endif

#define R__HAS_SETRESUID   /**/
#undef R__HAS_MATHMORE   /**/
#define R__HAS_PTHREAD    /**/
#undef R__HAS_XFT    /**/
#undef R__HAS_COCOA    /**/
#undef R__HAS_VC    /**/
#undef R__HAS_VDT    /**/
#undef R__HAS_VECCORE    /**/
#define R__USE_CXXMODULES   /**/
#undef R__USE_LIBCXX    /**/
#define R__HAS_ATTRIBUTE_ALWAYS_INLINE /**/
#define R__HAS_ATTRIBUTE_NOINLINE /**/
#undef R__USE_IMT   /**/
#undef R__COMPLETE_MEM_TERMINATION /**/
#undef R__HAS_CEFWEB  /**/
#undef R__HAS_QT6WEB  /**/
#undef R__HAS_DAVIX  /**/
#define R__HAS_DATAFRAME /**/
#define R__HAS_ROOT7 /**/
#undef R__LESS_INCLUDES /**/
#undef R__HAS_TBB /**/
#define R__HARDWARE_INTERFERENCE_SIZE 64
 /*Determined at CMake configure to be stable across all TUs*/

#if defined(R__HAS_VECCORE) && defined(R__HAS_VC)
#ifndef VECCORE_ENABLE_VC
#define VECCORE_ENABLE_VC
#endif
#endif

#undef R__HAS_DEFAULT_LZ4  /**/
#define R__HAS_DEFAULT_ZLIB  /**/
#undef R__HAS_DEFAULT_LZMA  /**/
#undef R__HAS_DEFAULT_ZSTD  /**/
#undef R__HAS_CLOUDFLARE_ZLIB /**/

#undef R__HAS_TMVACPU /**/
#undef R__HAS_TMVAGPU /**/
#define R__HAS_CUDNN /**/
#undef R__HAS_PYMVA /**/
#undef R__HAS_RMVA /**/

#undef R__HAS_URING /**/

#define R__HAS_GEOM /**/

#endif
