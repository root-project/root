/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * CINT header file G__ci.h
 ************************************************************************
 * Description:
 *  C/C++ interpreter header file
 ************************************************************************
 * Copyright(c) 1995~2007  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__CI_H
#define G__CI_H

#define FIXMESTRING2(x) #x
#define FIXMESTRING(x) FIXMESTRING2(x)
#define FIXME(TXT) __FILE__ "(" FIXMESTRING(__LINE__) "): FIXME! " TXT

#ifdef __cplusplus
//namespace Reflex;
//namespace ROOT { using namespace Reflex; }
#endif /* __cplusplus */

#ifndef G__CINT_VER6
#define G__CINT_VER6  1
#endif

#define G__CINTVERSION_BC     60030000
#define G__CINTVERSIONSTR_BC  "6.3.00, December 21, 2008"
#define G__CINTVERSION        70030000
#define G__CINTVERSIONSTR     "7.3.00, December 21, 2008"

#define G__ALWAYS
/* #define G__NEVER */
/**********************************************************************
* SPECIAL CHANGES and CINT CORE COMPILATION SWITCH
**********************************************************************/

#define G__NATIVELONGLONG 1

#ifndef G__CINT_VER6
#define G__OLDIMPLEMENTATION2187
#define G__OLDIMPLEMENTATION2184
#define G__OLDIMPLEMENTATION2182
#define G__OLDIMPLEMENTATION2177
#define G__OLDIMPLEMENTATION2172
#define G__OLDIMPLEMENTATION2171
#define G__OLDIMPLEMENTATION2170
#define G__OLDIMPLEMENTATION2169
#define G__OLDIMPLEMENTATION2163
#define G__OLDIMPLEMENTATION2162
#define G__OLDIMPLEMENTATION2161
#define G__OLDIMPLEMENTATION2160
#define G__OLDIMPLEMENTATION2159
#define G__OLDIMPLEMENTATION2156
#define G__OLDIMPLEMENTATION2155
#define G__OLDIMPLEMENTATION2154
#define G__OLDIMPLEMENTATION2153
#define G__OLDIMPLEMENTATION2152
#define G__OLDIMPLEMENTATION2151
#define G__OLDIMPLEMENTATION2150
#define G__OLDIMPLEMENTATION2148
#define G__OLDIMPLEMENTATION2147
#define G__OLDIMPLEMENTATION2146
#define G__OLDIMPLEMENTATION2143
#define G__OLDIMPLEMENTATION2142
#define G__OLDIMPLEMENTATION2141
#define G__OLDIMPLEMENTATION2140
#define G__OLDIMPLEMENTATION2138
#define G__OLDIMPLEMENTATION2137
#define G__OLDIMPLEMENTATION2136
#define G__OLDIMPLEMENTATION2135
#define G__OLDIMPLEMENTATION2134
#define G__OLDIMPLEMENTATION2133
#define G__OLDIMPLEMENTATION2132
#define G__OLDIMPLEMENTATION2131
#define G__OLDIMPLEMENTATION2129
#define G__OLDIMPLEMENTATION2128
#define G__OLDIMPLEMENTATION2127
#define G__OLDIMPLEMENTATION2122
#define G__OLDIMPLEMENTATION2117
#define G__OLDIMPLEMENTATION2116
/* #define G__OLDIMPLEMENTATION2115 */
/* #define G__OLDIMPLEMENTATION2114 */
#define G__OLDIMPLEMENTATION2112
#define G__OLDIMPLEMENTATION2111
#define G__OLDIMPLEMENTATION2110
#define G__OLDIMPLEMENTATION2109
#define G__OLDIMPLEMENTATION2105
#define G__OLDIMPLEMENTATION2102
#define G__OLDIMPLEMENTATION2089
#define G__OLDIMPLEMENTATION2087
#define G__OLDIMPLEMENTATION2084
#define G__OLDIMPLEMENTATION2075
#define G__OLDIMPLEMENTATION2074
#define G__OLDIMPLEMENTATION2073
#define G__OLDIMPLEMENTATION2067
#define G__OLDIMPLEMENTATION2066
#define G__OLDIMPLEMENTATION2062
#define G__OLDIMPLEMENTATION2058
/* #define G__OLDIMPLEMENTATION2057 */
/* #define G__OLDIMPLEMENTATION2056 */
#define G__OLDIMPLEMENTATION2054
#define G__OLDIMPLEMENTATION2051
#define G__OLDIMPLEMENTATION2042
#define G__OLDIMPLEMENTATION1073
#endif

#ifdef G__ROOT
/* Disable the new stack variable manager */
#define G__OLDIMPLEMENTATION1073
#endif

/* Native long long, unsigned long long, long double implementation */
#ifndef G__NATIVELONGLONG
#define G__OLDIMPLEMENTATION2189
#define G__OLDIMPLEMENTATION2192
#endif

/* Problem remains with autoloading if library is unloaded. Tried to fix it
 * with 2015, but this has problem with ROOT. */
#define G__OLDIMPLEMENTATION2015 


/* If you have problem compiling dictionary with static member function,
 * define following macro. */
/* #define G__OLDIMPLEMENTATION1993 */

/* 1987 fixes the same problem. Turned off because of redundancy. */
#define G__OLDIMPLEMENTATION1986

/* suppress unused parameter warnings. optional */
#ifndef G__SUPPRESS_UNUSEDPARA
#define G__OLDIMPLEMENTATION1911
#endif

/* &a, avoid uninitialized memory access */
/* #define G__AVOID_PTR_UNINITACCESS */  /* Turned out this fix was wrong */
#ifndef G__AVOID_PTR_UNINITACCESS
#define G__OLDIMPLEMENTATION1942
#endif

/* 1885 has side-effect in building ROOT */
#define G__OLDIMPLEMENTATION1885

/* 1770 changes implementation of skipping function implementation during
 * prerun. In order to activate new implementation, comment out following
 * line */
#define G__OLDIMPLEMENTATION1770


/* Change 1706, regarding function overriding, is very risky. So, this is
 * deactivated for now. With this change turned on, loading and unloading 
 * of interpreted and compiled function can be done more robustly. */
#define G__OLDIMPLEMENTATION1706

/* Rootcint's default link status has been changed from 5.15.57. 
 * Define following macro if new scheme has problems. */
/* #define G__OLDIMPLEMENTATION1700 */

/* For a machine which has unaddressable bool */
#ifndef G__UNADDRESSABLEBOOL
#if defined(__APPLE__) && defined(__ppc__)
/* Fons, if you find problems, comment out G__BOOL4BYTE and uncomment
 * G__UNADDRESSABLEBOOL. Thanks */
#define G__BOOL4BYTE
/* #define G__UNADDRESSABLEBOOL */
#endif
#endif

/* Activate pointer to member function handling in interpreted code. 
 * Seamless access of pointer to member between interpreted and compiled code
 * is not implemented yet. */
#ifndef G__PTR2MEMFUNC
#define G__PTR2MEMFUNC
#endif

/* Define following macro in order to disable iostream I/O redirection */
/* #define G__OLDIMPLEMENTATION1635 */

/* Define following macro to enable multi-thread safe libcint and DLL
 * features. */
/* #define G__MULTITHREADLIBCINT */

/* Define G__ERRORCALLBACK to activat error message redirection. If
 * G__ERRORCALLBACK is defined, a user can set a callback routine for
 * handling error message by G__set_errmsgcallback() API */
#ifndef G__ERRORCALLBACK
#define G__ERRORCALLBACK
#endif

/* 2001 masks G__ateval overloading resolution error. It turns out this is
 * not a good way, the feature is turned off */
#define G__OLDIMPLEMENTATION2001

/* Define following macros if you want to store where global variables
 * and typedefs are defined in source files. Reason of not making this
 * default is because it breaks DLL compatibility. */
#define G__VARIABLEFPOS
#define G__TYPEDEFFPOS 

/* If you use old g++ and having problem compiling dictionary with 
 * true pointer to function with const return value, define following
 * macro to workaround the problem. */
/* #define G__OLDIMPLEMENTATION1328 */

/* Define G__CONSTNESSFLAG for activating function overloading by
 * object constness. */
#define G__CONSTNESSFLAG
#ifndef G__CONSTNESSFLAG
#define G__OLDIMPLEMENTATION1258 /* create func entry w/wo func constness */
#define G__OLDIMPLEMENTATION1259 /* add isconst in G__value and set it */
#define G__OLDIMPLEMENTATION1260 /* use isconst info for func overloading */
#endif

/* New function overloading resolution algorithm which is closer to 
 * ANSI/ISO standard is implemented from cint5.14.35. This is a major
 * change and there are some risks. Define following macro in order to
 * use old algorithm. */
/* #define G__OLDIMPLEMENTATION1290 */

/* Define G__EXCEPTIONWRAPPER for activating C++ exception catching 
 * when calling precompiled function. It is better to define this macro
 * in platform dependency file OTHMACRO flag. Reason of not making this
 * default is because some old compilers may not support exception. */
/* #define G__EXCEPTIONWRAPPER */

/* Define G__STD_EXCEPTION for using std::exception in exception handler. 
 * If G__STD_EXCEPTION is defined, G__EXCEPTIONWRAPPER is also defined. */
/* #define G__STD_EXCEPTION */

/* If you define G__REFCONV in platform dependency file, bug fix for 
 * reference argument conversion is activated. This macro breaks DLL
 * compatibility between cint5.14.14 and 5.14.15. If you define
 * G__REFCONV, cint5.14.15 or newer version can load older DLL. But 
 * cint5.14.14 or older version can not load DLL that is created by
 * cint5.14.15 or later cint. */
#define G__REFCONV

/* This change activates bytecode compilation of class object 
 * instantiation in a function. Because the change includes some
 * problems , it is turned off at this moment by defining following
 * macro. */
#ifdef G__OLDIMPLEMENTATION1073
/* define related macros here */
#endif

/* Scott Snyder's modification Apr1999 9.Improvements for `variable' macros. 
 * Comment out line below to activate the change */
#define G__OLDIMPLEMENTATION1062

/* Scott Snyder's modification Apr1999 10.More CRLF problems 
 * Comment out line below to activate the change */
#define G__OLDIMPLEMENTATION1063

/* Scott Snyder's modification in macro.c around line 709. Apr1999
 * Uncomment following line to use 969 version */
/* #define G__OLDIMPLEMENTATION973 */


/* Unlimited number of function arguments. THIS MODIFICATION IS TURNED OFF
 * because the change did not work. I decided to keep the code somehow. */
#define G__OLDIMPLEMENTATION834 

/**********************************************************************
* END OF SPECIAL CHANGES and CINT CORE COMPILATION SWITCH
**********************************************************************/

/**************************************************************************
* One of following macro has to be defined to fix DLL global function
* conflict problem. G__CPPIF_STATIC is recommended. Define G__CPPIF_PROJNAME
* only if G__CPPIF_STATIC has problem with your compiler.
**************************************************************************/
#ifdef G__CPPIF_EXTERNC
#ifndef G__CPPIF_PROJNAME
#define G__CPPIF_PROJNAME
#endif
#ifdef G__CPPIF_STATIC
#undef G__CPPIF_STATIC
#endif
#endif

#ifndef G__CPPIF_PROJNAME
#ifndef G__CPPIF_STATIC
#define G__CPPIF_STATIC
#endif
#endif

/**************************************************************************
* G__reftype, var->reftype[], ifunc->reftype[] flag
**************************************************************************/
#define G__PARANORMAL       0
#define G__PARAREFERENCE    1
#define G__PARAP2P          2
#define G__PARAP2P2P        3

#define G__PARAREF         100
#define G__PARAREFP2P      102
#define G__PARAREFP2P2P    103

/**************************************************************************
* if __MAKECINT__ is defined, do not include this file
* G__MAKECINT is automatically defined in makecint or G__makesetup script
**************************************************************************/
#if (!defined(__MAKECINT__)) || defined(G__API)


#ifdef __cplusplus
#ifndef G__ANSIHEADER
#define G__ANSIHEADER
#endif
#endif

#ifdef __SC__
#ifndef G__SYMANTEC
#define G__SYMANTEC
#endif
#endif

#ifdef __QNX__
#ifndef G__QNX
#define G__QNX
#endif
#endif

#ifdef _MSC_VER
#ifndef G__VISUAL
#define G__VISUAL 1
#endif
#ifndef G__MSC_VER
#define G__MSC_VER
#endif
#endif

#ifdef __VMS
#define G__VMS
#endif

#if defined(__BORLANDC__) || defined(__BCPLUSPLUS) || defined(__BCPLUSPLUS__) || defined(G__BORLANDCC5)
#ifndef G__BORLAND
#define G__BORLAND
#endif
#endif

#ifdef G__BORLANDCC5
#define G__SHAREDLIB
#define G__DLL_SYM_UNDERSCORE
#define G__WIN32
#define G__ANSI
#define G__P2FCAST
#define G__REDIRECTIO
#define G__DETECT_NEWDEL
#define G__POSIX
#define G__STD_EXCEPTION
#endif

#if defined(_WIN32) || defined(_WINDOWS) || defined(_Windows) || defined(_WINDOWS_)
#ifndef G__WIN32
#define G__WIN32
#endif
#endif
  
/* added by Fons Radamakers in 2000 Oct 2 */
#if (defined(__linux) || defined(__linux__) || defined(linux)) && ! defined(__CINT__)
#   include <features.h>
#   if __GLIBC__ == 2 && __GLIBC_MINOR__ >= 2
#      define G__NONSCALARFPOS2
#   endif
#endif

/***********************************************************************
 * Native long long support
 ***********************************************************************/
#if defined(G__WIN32) && !defined(__CINT__)
typedef __int64            G__int64;
typedef unsigned __int64   G__uint64;
#else
typedef long long          G__int64;
typedef unsigned long long G__uint64;
#endif


/***********************************************************************
 * Something that depends on platform
 ***********************************************************************/

#define ENABLE_CPP_EXCEPTIONS 1

/* Exception */
#if defined(G__WIN32) && !defined(G__STD_EXCEPTION)
#define G__STD_EXCEPTION
#endif
#if defined(G__STD_EXCEPTION) && !defined(G__EXCEPTIONWRAPPER)
#define G__EXCEPTIONWRAPPER
#endif

/* Error redirection ,  G__fprinterr */
#if defined(G__WIN32) && !defined(G__ERRORCALLBACK)
#define G__ERRORCALLBACK
#endif
#ifndef G__ERRORCALLBACK
#define G__OLDIMPLEMENTATION1485
#define G__OLDIMPLEMENTATION2000
#endif

/* temporary file generation */
#if defined(G__WIN32) 
#define G__TMPFILE
#endif


/***********************************************************************
 * Define G__EH_DUMMY_DELETE in order to avoid some compiler dependency
 * about 'void operator delete(void*,[DLLID]_tag*);'
 ***********************************************************************/
#if defined(__HP_aCC) || defined(G__VISUAL)
#define G__EH_DUMMY_DELETE
#endif

#ifdef __CINT__
#undef G__WIN32
#endif

#ifdef G__NONANSI
#ifdef G__ANSIHEADER
#undef G__ANSIHEADER
#endif
#endif

#ifndef G__IF_DUMMY
#define G__IF_DUMMY /* avoid compiler warning */
#endif

#if defined(G__BORLAND) || defined(G__VISUAL)
#define G__DLLEXPORT __declspec(dllexport)
#define G__DLLIMPORT __declspec(dllimport)
#else
#define G__DLLEXPORT
#define G__DLLIMPORT
#endif

#if (defined(G__BORLAND)||defined(G__VISUAL)||defined(G__CYGWIN)) && defined(G__CINTBODY) && !defined(__CINT__)
#define G__EXPORT __declspec(dllexport)
#else
#define G__EXPORT
#endif


#if defined(G__SIGNEDCHAR) 
typedef signed char G__SIGNEDCHAR_T;
#else
typedef char G__SIGNEDCHAR_T;
#endif

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <signal.h>
#include <assert.h>
#include <limits.h>
#include <setjmp.h>
/* #include <time.h> */
#include <ctype.h>
#include <fcntl.h>

#ifndef G__WIN32
#include <unistd.h>
#endif

#ifdef G__REGEXP
#include <regex.h>
#endif

#ifdef G__REGEXP1
#include <libgen.h>
#endif

#if   defined(G__SUNOS4)
#include "src/sunos.h"
#elif defined(G__NEWSOS4) || defined(G__NEWSOS6)
#include "src/newsos.h"
#elif defined(G__NONANSI)
#include "src/sunos.h"
#endif

#define G__DUMPFILE
#define G__DOSHUGE


#ifndef G__REFCONV
#define G__OLDIMPLEMENTATION1167
#endif


/* Special typeinfo enhacement for Fons Rademaker's request */
#define G__FONS_TYPEINFO
#define G__FONS_COMMENT
#define G__FONS_ROOTSPECIAL
#define G__ROOTSPECIAL

/**********************************************************************
* Function call stack
**********************************************************************/
#define G__SHOWSTACK
#define G__VAARG

/**************************************************************************
* Dump function calls to '-d [dumpfile]', if G__DUMPFILE is defined.
*
**************************************************************************/

/**************************************************************************
* Interpreter Security mode
*
**************************************************************************/
#ifndef G__NOSECURITY
#define G__SECURITY
#endif

#include "G__security.h"

/**************************************************************************
* True pointer to global function
*
**************************************************************************/
#define G__TRUEP2F

/**************************************************************************
* Whole function compilation
*
**************************************************************************/
#define G__ASM_FUNC
#define G__ASM_WHOLEFUNC

/**************************************************************************
* C++  evolution has begun from revision 3.0.10.
*
* Define macro 'G__CPLUSPLUS' for C++ version.
* If G__CPLUSPLUS is not defined, all C++ features are turned off. In this
* case it must be compatible with 3.0.9.
**************************************************************************/
#define G__CPLUSPLUS



#ifdef G__CPLUSPLUS

/**********************************************************************
* Object oriented feature of C++
**********************************************************************/

/* Create default assignment operator for new C++ linkage */
/* #define G__DEFAULTASSIGNOPR */

/* virtual base class */
#define G__VIRTUALBASE

/* new inheritance implementation */
#define G__NEWINHERIT

/* Friend class and function */
#define G__FRIEND

/* Run time type information */
#define G__TYPEINFO

/* new, delete operator */
#define G__NEWDELETE
#define G__NEWDELETE_YET

/* destructor */
#define G__DESTRUCTOR

/* constructor */
#define G__CONSTRUCTOR
#define G__COPYCONSTRUCTOR

/* member function */
#define G__MEMBERFUNC

/* keyword class */
#define G__CLASS

/* member access control */
#define G__ACCESS

#ifdef G__NEWINHERIT
#define G__PUBLIC       0x01
#define G__PROTECTED    0x02
#define G__PRIVATE      0x04
#define G__GRANDPRIVATE 0x08
#define G__PUBLIC_PROTECTED_PRIVATE 0x7
#define G__PUBLIC_PROTECTED         0x3

#else
#define G__PUBLIC    0
#define G__PROTECTED 1
#define G__PRIVATE   2
#define G__GRANDPRIVATE 3
#endif

/* inheritance */
#define G__INHERIT
#define G__INHERIT1
#define G__INHERIT2
#define G__INHERIT3
#define G__INHERIT4
#define G__INHERIT5

#define G__EXPLICITCONV

#ifdef __CINT__
typedef int (*G__IgnoreInclude)();
#endif

/**********************************************************************
* Non object oriented feature of C++
**********************************************************************/

/***************************************************************
* Implementation of function/operator overloading is not
* completed. It is very premature.
***************************************************************/

/* if G__OVERLOADOPERATOR is defined, G__OVERLOADFUNC must be also defined */
#define G__OVERLOADOPERATOR
#define G__OVERLOADOPERATOR2

/* if G__OVERLOADFUNC is defined, G__IFUNCPARA must be also defined */
#define G__OVERLOADFUNC

#define G__OVERLOADFUNC2
#define G__EXACT     1
#define G__PROMOTION 2
#define G__STDCONV   3
#define G__USERCONV  4

/* for struct,class,union return value */
#define G__TEMPOBJECT
#define G__TEMPOBJECT2

/* reference type */
#define G__REFERENCETYPE

/* improved version of reference type implementation */
#define G__REFERENCETYPE2

/***************************************************************
* Having default parameter for function
***************************************************************/

/* G__DEFAULTPARAMETER can be defined independently */
#define G__DEFAULTPARAMETER


/***************************************************************
* reading and storing parameter type for ANSI stype function
* header. This functionality itself can be added to non C++
* version but it won't play essential part.  For C++ version,
* function parameter information is needed for function/operator
* overloading.
***************************************************************/

/* G__IFUNCPARA can be defined independently */
#define G__IFUNCPARA

/* C++ object linkage */
#define G__CPPSTUB     5
#define G__CPPLINK    -1
#define G__CPPLINK1
#define G__CPPLINK2
#define G__CPPLINK3

/* C object linkage same way as C++ */
#define G__CSTUB       6
#define G__CLINK      -2

/* define for Reflex cpp source code */
#define R__CPPLINK  -3

/* Link macro as function */
#define G__MACROLINK  (-5)

/* Link macro as function */
#define G__METHODLINK  (-6)
#define G__ONLYMETHODLINK  6

#define G__NOLINK      0



#else /* of G__CPLUSPLUS */

/***************************************************************
* new comment style   //
***************************************************************/
#define G__NOCPPCOMMENT

#endif /* of G__CPLUSPLUS */

/**************************************************************************
* Table and variable size
*
* CAUTION:
*  Among constants below, changing following parameter cause DLL binary
* incompatibility. 
*
*    G__MAXFUNCPARA
*
* Other parameters can be changed while keeping DLL binary compatibility.
*
**************************************************************************/
#ifdef G__LONGBUF
#define G__LONGLINE    4096  /* Length of expression */
#define G__ONELINE     4096  /* Length of subexpression,parameter,argument */
#define G__ONELINEDICT    8  /* Length of subexpression,parameter,argument */
#define G__MAXNAME     4096  /* Variable name */
#else
#define G__LONGLINE    2048  /* Length of expression */
#define G__ONELINE     1024  /* Length of subexpression,parameter,argument */
#define G__MAXNAME      512  /* Variable name */
#define G__ONELINEDICT    8  /* Length of subexpression,parameter,argument */
#endif
#define G__LARGEBUF    6000  /* big temp buffer */
#define G__MAXFILE     2000  /* Max interpreted source file */
#define G__MAXFILENAME 1024  /* Max interpreted source file name length */
#define G__MAXPARA      100  /* Number of argument for G__main(argc,argv)   */
#define G__MAXARG       100  /* Number of argument for G__init_cint(char *) */
#define G__MAXFUNCPARA   40  /* Function argument */
#define G__MAXVARDIM     10  /* Array dimension */
#define G__LENPOST       10  /* length of file name extention */
#define G__MAXBASE       50  /* maximum inheritable class */
#define G__TAGNEST       20  /* depth of nested class */

#ifdef G__WIN32
#if defined(_MSC_VER) && (_MSC_VER>1300)
#define G__MAXSTRUCT  24000  /* struct table */
#define G__MAXTYPEDEF 24000  /* typedef table */
#else
#define G__MAXSTRUCT   4000  /* struct table */
#define G__MAXTYPEDEF  8000  /* typedef table */
#endif
#else
#define G__MAXSTRUCT  24000  /* struct table */
#define G__MAXTYPEDEF 24000  /* typedef table */
#endif

/* G__MAXIFUNC and G__MEMDEPTH are not real limit
 * They are depth of one page of function or variable list
 * If the page gets full, more table is allocated. */
#define G__MAXIFUNC 1
#define G__MEMDEPTH 1


/* #define G__HIST     1 */

/**************************************************************************
* error handling
**************************************************************************/
#define G__TIMEOUT 10   /* Timeout after segv,buserror,etc */

/**************************************************************************
* variable identity
**************************************************************************/
#define G__AUTO (-1)
#define G__LOCALSTATIC (-2)
#define G__LOCALSTATICBODY (-3)
#define G__COMPILEDGLOBAL  (-4)
#define G__AUTOARYDISCRETEOBJ (-5)

#define G__LOCAL    0
#ifdef G__MEMBERFUNC
#define G__MEMBER   2
#define G__GLOBAL   4
#define G__NOTHING  6
#else
#define G__GLOBAL   2
#endif




#ifdef __cplusplus
extern "C" {
#endif

/**************************************************************************
* struct of pointer to pointer flag
*
* By histrorical reason, cint handles pointer to pointer in following manner.
*
* islower(buf.type)&&G__PARANORMAL==buf.obj.reftype.reftype :object
* isupper(buf.type)&&G__PARANORMAL==buf.obj.reftype.reftype :pointer to object
* isupper(buf.type)&&G__PARAP2P==buf.obj.reftype.reftype    :pointer to pointer
* isupper(buf.type)&&G__PARAP2PP2==buf.obj.reftype.reftype  :pointer to pointer
*                                                            to pointer
**************************************************************************/
struct G__p2p {
   /* need to keep to not change sizeof(G__value) during merge*/
  long i_OLD_DONT_USE;
  int reftype_OLD_DONT_USE;
};

/**************************************************************************
* struct of internal data
*
**************************************************************************/
#ifdef __cplusplus
namespace Reflex {
   class TypeName;
}
#endif
struct G__REFLEXTYPE_C_PLACEHOLDER {
#ifdef __cplusplus
   Reflex::TypeName* fTypeName;
#else
   void* fTypeName;
#endif
   unsigned int fModifiers;
};
struct G__DUMMY_FOR_CINT5 {
   // Stuff we removed from Cint5
   int type;
   int tagnum;
   int typenum;
#ifndef G__OLDIMPLEMENTATION1259
   G__SIGNEDCHAR_T isconst;
#endif
};
#ifdef __cplusplus
struct G__value {
#else
typedef struct {
#endif
  union {
    double d;
    long    i; /* used to be int */
    struct G__p2p reftype;
    char ch;
    short sh;
    int in;
    float fl;
    unsigned char uch;
    unsigned short ush;
    unsigned int uin;
    unsigned long ulo;
    G__int64 ll;
    G__uint64 ull;
    long double ld;
  } obj;
#ifdef G__REFERENCETYPE2
  long ref;
#endif
  struct G__DUMMY_FOR_CINT5 dummyFotCint5;
  struct G__REFLEXTYPE_C_PLACEHOLDER buf_typenum;
}
#ifdef __cplusplus

#else
  G__value
#endif
;  

/**************************************************************************
* structure for function and array parameter
*
**************************************************************************/
struct G__param {
  int paran;
#ifdef G__OLDIMPLEMENTATION1530
  char parameter[G__MAXFUNCPARA][G__ONELINE];
#endif
  G__value para[G__MAXFUNCPARA];
#ifndef G__OLDIMPLEMENTATION1530
  char parameter[G__MAXFUNCPARA][G__ONELINE];
#endif
};

/**************************************************************************
* Compiled class tagnum table
*
**************************************************************************/
typedef struct {
#ifdef __cplusplus
  const char *tagname;
#else
  char *tagname;
#endif
  char tagtype;
  short tagnum;
} G__linked_taginfo;

/**************************************************************************
* reference type argument for precompiled function
**************************************************************************/
#define G__Mfloat(buf)   (buf.obj.fl=(float)G__double(buf))
#define G__Mdouble(buf)  buf.obj.d
#define G__Mchar(buf)    (buf.obj.ch=(char)buf.obj.i)
#define G__Mshort(buf)   (buf.obj.sh=(short)buf.obj.i)
#define G__Mint(buf)     (buf.obj.in=(int)buf.obj.i)
#define G__Mlong(buf)    buf.obj.i
#define G__Muchar(buf)   (buf.obj.uch=(unsigned char)buf.obj.i)
#define G__Mushort(buf)  (buf.obj.ush=(unsigned short)buf.obj.i)
#define G__Muint(buf)    (*(unsigned int*)(&buf.obj.i))
#define G__Mulong(buf)   (*(unsigned long*)(&buf.obj.i))



/**************************************************************************
* include file flags
**************************************************************************/
#define G__USERHEADER 1
#define G__SYSHEADER  2


#ifndef G__ANSI
#if (__GNUC__>=3) || defined(_STLPORT_VERSION)
#define G__ANSI
#endif
#endif
/* #define G__ANSI */

#ifdef __cplusplus

#ifndef G__ANSI
#define G__ANSI
#endif
#ifndef __CINT__
#define G__CONST const
#else
#define G__CONST 
#endif

#else /* __cplusplus */

#define G__CONST 

#endif /* __cplusplus */

extern G__value G__null;

/**************************************************************************
* struct for variable page buffer
*
**************************************************************************/
#ifndef __CINT__
#define G__VARSIZE  2
#define G__CHARALLOC   sizeof(char)
#define G__SHORTALLOC  sizeof(short)
#define G__INTALLOC    sizeof(int)
#define G__LONGALLOC   sizeof(long)
#define G__FLOATALLOC  sizeof(float)
#define G__DOUBLEALLOC sizeof(double)
#define G__P2MFALLOC   G__sizep2memfunc
#define G__LONGLONGALLOC sizeof(G__int64)
#define G__LONGDOUBLEALLOC sizeof(long double)
#endif /* __CINT__ */

#ifdef G__TESTMAIN
/* This is only needed for demonstration that cint interprets cint */
#define G__VARSIZE  2
#define G__CHARALLOC   sizeof(char)
#define G__SHORTALLOC  sizeof(short)
#define G__INTALLOC    sizeof(int)
#define G__LONGALLOC   sizeof(long)
#define G__FLOATALLOC  sizeof(float)
#define G__DOUBLEALLOC sizeof(double)
#define G__P2MFALLOC   G__sizep2memfunc
#endif

/**************************************************************************
* CINT API function return value
*
**************************************************************************/
/* return value of G__init_cint() */
#define G__INIT_CINT_FAILURE         (-1)
#define G__INIT_CINT_SUCCESS          0
#define G__INIT_CINT_SUCCESS_MAIN     1

/* return value of G__loadfile() */
#define G__LOADFILE_SUCCESS         0
#define G__LOADFILE_DUPLICATE       1
#define G__LOADFILE_FAILURE       (-1)
#define G__LOADFILE_FATAL         (-2)

/* return value of G__unloadfile() */
#define G__UNLOADFILE_SUCCESS    0
#define G__UNLOADFILE_FAILURE  (-1)

/* return value of G__pause() */
#define G__PAUSE_NORMAL          0
#define G__PAUSE_IGNORE          1
#define G__PAUSE_STEPOVER        3
#define G__PAUSE_ERROR_OFFSET 0x10

/* return value of G__interpretedp2f() */
#define G__NOSUCHFUNC              0
#define G__UNKNOWNFUNC             0
#define G__INTERPRETEDFUNC         1
#define G__COMPILEDWRAPPERFUNC     2
#define G__COMPILEDINTERFACEMETHOD 2
#define G__COMPILEDTRUEFUNC        3
#define G__BYTECODEFUNC            4

/* flags to set to G__ismain */
#define G__NOMAIN                  0
#define G__MAINEXIST               1
#define G__TCLMAIN                 2

/*********************************************************************
* return status flag
*********************************************************************/
#define G__RETURN_NON       0
#define G__RETURN_NORMAL    1 
#define G__RETURN_IMMEDIATE 2
#define G__RETURN_TRY      -1 
#define G__RETURN_EXIT1     4
#define G__RETURN_EXIT2     5

/**************************************************************************
* struct forward declaration; real ones are in common.h
**************************************************************************/
struct G__var_array;
struct G__dictposition; // decl in Api.h because of C++ content

struct G__comment_info;
struct G__friendtag;
#ifdef G__ASM_WHOLEFUNC
struct G__bytecodefunc;
#endif
struct G__RootSpecial;
struct G__inheritance;
struct G__tagtable;
struct G__input_file;
#ifdef G__CLINK
struct G__tempobject_list;
#endif
struct G__va_list_para;

/**************************************************************************
* flag argument to G__getfunction()
**************************************************************************/
#define G__TRYNORMAL         0
#define G__CALLMEMFUNC       1
#define G__TRYMEMFUNC        2
#define G__CALLCONSTRUCTOR   3  
#define G__TRYCONSTRUCTOR    4  
#define G__TRYDESTRUCTOR     5  
#define G__CALLSTATICMEMFUNC 6
#define G__TRYUNARYOPR       7
#define G__TRYBINARYOPR      8

#ifndef G__OLDIMPLEMENTATINO1250
#define G__TRYIMPLICITCONSTRUCTOR 7
#endif

/**************************************************************************
* Scope operator category
**************************************************************************/
#define G__NOSCOPEOPR    0
#define G__GLOBALSCOPE   1
#define G__CLASSSCOPE    2

/*********************************************************************
* variable length string buffer
*********************************************************************/
#define G__LONGLONG    1
#define G__ULONGLONG   2
#define G__LONGDOUBLE  3

/**************************************************************************
* store environment for stub function casll
**************************************************************************/
struct G__StoreEnv {
  long store_struct_offset;
  int store_tagnum;
  int store_memberfunc_tagnum;
  int store_exec_memberfunc;
};

/**************************************************************************
 * Variable argument, byte layout policy
 **************************************************************************/
#define G__VAARG_SIZE 1024

typedef struct G__va_arg_buf_TAG {
  union {
    char d[G__VAARG_SIZE];
    long i[G__VAARG_SIZE/sizeof(long)];
  } x;
} G__va_arg_buf;

#define G__NOSTREAMER      0x01
#define G__NOINPUTOPERATOR 0x02
#define G__USEBYTECOUNT    0x04
#define G__HASVERSION      0x08

#define G__BYTECODE_NOTYET    1
#define G__BYTECODE_FAILURE   2
#define G__BYTECODE_SUCCESS   3
#define G__BYTECODE_ANALYSIS  4 /* ON1164 */

/**************************************************************************
* structure for input file
*
**************************************************************************/
struct G__input_file {
  FILE *fp;
  int line_number;
  short filenum;
  char name[G__MAXFILENAME];
#ifndef G__OLDIMPLEMENTATION1649
  char *str;
  unsigned long pos;
  int vindex;
#endif
};

/************************************************************************** 	 
* structure for friend function and class 	 
* 	 
**************************************************************************/ 	 
struct G__friendtag { 	 
  short tagnum; 	 
  struct G__friendtag *next; 	 
};

/**************************************************************************
* Interface Method type
*
**************************************************************************/
#if defined(G__ANSIHEADER) || defined(G__ANSI)
typedef int (*G__InterfaceMethod)(G__value*,G__CONST char*,struct G__param*,int);
#else
typedef int (*G__InterfaceMethod)();
#endif

#ifdef __cplusplus
typedef void (*G__incsetup)(void);
#else  /* __cplusplus */
typedef void (*G__incsetup)();
#endif /* __cplusplus */

#define G__ISDIRECTINHERIT         0x0001
#define G__ISVIRTUALBASE           0x0002
#define G__ISINDIRECTVIRTUALBASE   0x0004

/**************************************************************************
* make hash value
*
**************************************************************************/

#define G__hash(string,hash,len) len=hash=0;while(string[len]!='\0')hash+=string[len++];

#define G__TEMPLATEMEMFUNC
#ifdef G__TEMPLATEMEMFUNC
/* Doubly linked list of long int, methods are described in tmplt.c */
struct G__IntList {
  long i;
  struct G__IntList *prev;
  struct G__IntList *next;
};

struct G__Definedtemplatememfunc {
  int line;
  int filenum;
  FILE *def_fp;
  fpos_t def_pos;
  struct G__Definedtemplatememfunc *next;
};
#endif

struct G__Templatearg {
  int type;
  char *string;
  char *default_parameter;
  struct G__Templatearg *next;
};

struct G__Definedtemplateclass {
  char *name;
  int hash;
  int line;
  int filenum;
  FILE *def_fp;
  fpos_t def_pos;
  struct G__Templatearg *def_para;
#ifdef G__TEMPLATEMEMFUNC
  struct G__Definedtemplatememfunc memfunctmplt;
#endif
  struct G__Definedtemplateclass *next;
  int parent_tagnum;
  struct G__IntList *instantiatedtagnum;
  int isforwarddecl;
  int friendtagnum;
  struct G__Definedtemplateclass *specialization;
  struct G__Templatearg *spec_arg;
};

/********************************************************************
* include path by -I option
* Used in G__main() and G__loadfile()
********************************************************************/
struct G__includepath {
  char *pathname;
  struct G__includepath *next;
};

/**************************************************************************
* Forward declaration of Cint 5 structure for backward compatibility
**************************************************************************/
struct G__ifunc_table;

/**************************************************************************
* pointer to function which is evaluated at pause
**************************************************************************/
extern void (*G__atpause)();

/**************************************************************************
* pointer to function which is evaluated in G__genericerror()
**************************************************************************/
extern void (*G__aterror)();

/**************************************************************************
* New handling of pointer to function
*
**************************************************************************/
#ifndef G__NONANSI
#ifndef G__SUNOS4
#define G__FUNCPOINTER
#endif
#endif

/**************************************************************************
* Bug fix for struct allocation
*
**************************************************************************/
#define G__PVOID ((char*)(-1))
#define G__PINVALID 0

/**********************************************************************
* Multi-byte character handling in comment and string
**********************************************************************/
#define G__MULTIBYTE

/**************************************************************************
* ASSERTION MACRO
*
**************************************************************************/
#if defined(G__DEBUG) || defined(G__ASM_DBG)

#define G__ASSERT(f)                                                      \
  if(!(f)) fprintf(G__serr                                                \
                   ,"cint internal error: %s line %u FILE:%s LINE:%d\n"   \
                   ,__FILE__,__LINE__,G__ifile.name,G__ifile.line_number)


#else

#define G__ASSERT(f)  /* NULL */

#endif


#ifndef __CINT__

#if defined(G__DEBUG) && !defined(G__MEMTEST_C)
#include "src/memtest.h"
#endif

#ifndef G__WILDCARD
#define G__WILDCARD
#endif

extern G__EXPORT G__value (*G__GetSpecialObject) (char *name,void **ptr,void** ppdict);
extern G__EXPORT int (*G__ScriptCompiler) (G__CONST char*,G__CONST char*);
typedef int (*G__IgnoreInclude) (const char* fname,const char* expandedfname);
typedef void G__parse_hook_t ();

/**************************************************************************
* Exported Functions
*
**************************************************************************/
#ifndef G__DECL_API
# ifndef G__MULTITHREADLIBCINT
#  ifdef __cplusplus
#    define G__DUMMYTOCHECKFORDUPLICATES_CONCAT(A,B) A##B
#    define G__DUMMYTOCHECKFORDUPLICATES(IDX) namespace{class G__DUMMYTOCHECKFORDUPLICATES_CONCAT(this_API_function_index_occurs_more_than_once_,IDX) {};}
#  else
#    define G__DUMMYTOCHECKFORDUPLICATES(IDX) 
#  endif
#  define G__DECL_API(IDX, RET, NAME, ARGS) \
   G__EXPORT RET NAME ARGS ; G__DUMMYTOCHECKFORDUPLICATES(IDX)
# else
#  define G__DUMMYTOCHECKFORDUPLICATES(IDX)
#  define G__DECL_API(IDX, RET, NAME, ARGS) \
     static RET (* NAME ) ARGS = 0;
# endif
#endif /*G__DECL_API*/

#include "G__ci_fproto.h"

#ifdef G__MULTITHREADLIBCINT
/* second round, now setting func ptrs */

# undef  G__DUMMYTOCHECKFORDUPLICATES
# define G__DUMMYTOCHECKFORDUPLICATES(IDX)
# ifdef G__MULTITHREADLIBCINTC
#  define G__SET_CINT_API_POINTERS_FUNCNAME G__SetCCintApiPointers
# else
#  define G__SET_CINT_API_POINTERS_FUNCNAME G__SetCppCintApiPointers
# endif
# undef G__DECL_API
# define G__DECL_API(IDX, RET, NAME, ARGS) \
     NAME = (RET (*) ARGS) a[IDX];

G__EXPORT void G__SET_CINT_API_POINTERS_FUNCNAME (void *a[G__NUMBER_OF_API_FUNCTIONS]) {
#include "G__ci_fproto.h"
}
#endif /*G__MULTITHREADLIBCINT*/


/**************************************************************************
 * end of Exported Cint API functions
 **************************************************************************/

#endif /* __CINT__ */
   
#endif /* __MAKECINT__ */
/**************************************************************************
* endif #ifndef G__MAKECINT
**************************************************************************/

/* C interface */

typedef struct {
  char *name;
  void (*pfunc)();
} G__COMPLETIONLIST;

#ifdef __cplusplus
} /* extern "C" */
#endif

/***********************************************************************/
#if defined(__cplusplus) && !defined(__CINT__)
   // Helper class to avoid compiler warning about casting function pointer
   // to void pointer.
   class G__func2void {
      typedef void (*funcptr_t)();
      
      union funcptr_and_voidptr {
         typedef void (*funcptr_t)();
         funcptr_and_voidptr(void *val) : _read(val) {}         
         void *_read;
         funcptr_t _write;
      };
      
      funcptr_and_voidptr _tmp;
   public:
      template <typename T>
      G__func2void( T vfp ) : _tmp(0) {
         _tmp._write = ( funcptr_t )vfp;
      }
      
      operator void* () const {
         return _tmp._read;
      }
   };
#endif
   
#endif /* G__CI_H */

