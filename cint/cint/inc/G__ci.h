/* /% C %/ */
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

#ifndef G__CINT_VER6
#define G__CINT_VER6  1
#endif

#define G__CINTVERSION_V6      60020000
#define G__CINTVERSIONSTR_V6  "6.2.00, Dec 21, 2008"
#define G__CINTVERSION_V5      50170000
#define G__CINTVERSIONSTR_V5  "5.17.00, Dec 21, 2008"

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

/* Define G__FIX1 if you have problem defining variable argument functions
 * such as printf, fprintf, etc... in Windows */
/* #define G__FIX1 */

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

/* Speed up G__strip_quotation */
#ifdef G__ROOT
#ifndef G__CPPCONSTSTRING
#define G__CPPCONSTSTRING
#endif
#endif

/* Activate pointer to member function handling in interpreted code.
 * Seamless access of pointer to member between interpreted and compiled code
 * is not implemented yet. */
#ifndef G__PTR2MEMFUNC
#define G__PTR2MEMFUNC
#endif

/* 1649 is not ready yet */
/* #define G__OLDIMPLEMENTATION1649 */

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
#if (!defined(__MAKECINT__)) || defined(G__API) || defined(G__BC_DICT)


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
#if defined(G__STD_EXCEPTION) && !defined(G__EXCEPTIONWRAPPER) && !defined(G__APIIF)
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
#if defined(__HP_aCC) || defined(G__VISUAL) || defined(__INTEL_COMPILER)
#define G__EH_DUMMY_DELETE
#endif

#ifdef G__NONANSI
#ifdef G__ANSIHEADER
#undef G__ANSIHEADER
#endif
#endif

#ifndef G__IF_DUMMY
#define G__IF_DUMMY /* avoid compiler warning */
#endif

#ifdef G__VMS
#ifndef G__NONSCALARFPOS
#define G__NONSCALARFPOS
#endif
typedef long fpos_tt; /* pos_t is defined to be a struct{32,32} in VMS.
                         Therefore,pos_tt is defined to be a long. This
                         is used in G__ifunc_table_VMS, G__functentry_VMS*/
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



#if defined(__cplusplus) && !defined(__CINT__)
extern "C" {   /* extern C 1 */
#endif

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
#define G__SECURITY

#ifdef G__SECURITY

/* #include "include/security.h" */

#define G__NOERROR            0x0000
#define G__RECOVERABLE        0x0001
#define G__DANGEROUS          0x0002
#define G__FATAL              0x0004

#if (defined(G__SGICC) || defined(G__DECCXX)) && defined(G__ROOT)
#define G__CHECK(ITEM,COND,ACTION)                                       \
 if((G__security=G__SECURE_LEVEL0)&&(G__security&ITEM) && (COND) && G__security_handle(ITEM)) ACTION
#else
#define G__CHECK(ITEM,COND,ACTION)                                       \
 if((G__security&ITEM) && (COND) && G__security_handle(ITEM)) ACTION
#endif

#define G__CHECKDRANGE(p,low,up) \
 if(G__check_drange(p,low,up,G__double(libp->para[p]),result7,funcname)) \
   return(1)

#define G__CHECKLRANGE(p,low,up) \
 if(G__check_lrange(p,low,up,G__int(libp->para[p]),result7,funcname)) return(1)

#define G__CHECKTYPE(p,t1,t2) \
 if(G__check_type(p,t1,t2,&libp->para[p],result7,funcname)) return(1)

#define G__CHECKNONULL(p,t) \
 if(G__check_nonull(p,t,&libp->para[p],result7,funcname)) return(1)

#define G__CHECKNPARA(n) \
 if(n!=libp->paran) { \
    G__printerror(funcname,n,libp->paran); \
    *result7=G__null; \
    return(1); \
 }

#else /* G__SECURITY */

#define G__CHECK(ITEM,COND,ACTION)   NULL
#define G__CHECKDRANE(p,up,low)  NULL
#define G__CHECKLRANE(p,up,low)  NULL
#define G__CHECKNONULL(p)  NULL
#define G__CHECKNPARA(n)  NULL

#endif /* G__SECURITY */


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
*
**************************************************************************/
struct G__StoreEnv {
  long store_struct_offset;
  int store_tagnum;
  int store_memberfunc_tagnum;
  int store_exec_memberfunc;
};


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
  long i;
  int reftype;
};

/**************************************************************************
* struct of internal data
*
**************************************************************************/
struct G__DUMMY_FOR_CINT7 {
   void* fTypeName;
   unsigned int fModifiers;
};

#ifdef __cplusplus
struct G__value {
#else
typedef struct {
#endif
  union {
    double d;
    long    i; /* used to be int */
#if defined(G__PRIVATE_GVALUE) && !defined(_WIN32)
#if defined(private) && defined(ROOT_RVersion)
#define G__alt_private private
#undef private
#endif
private:
#endif
    struct G__p2p reftype;
#if defined(G__PRIVATE_GVALUE) && !defined(_WIN32)
public:
#endif
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
#if defined(G__PRIVATE_GVALUE) && !defined(_WIN32)
private:
#if defined(G__alt_private) && defined(ROOT_RVersion)
#define private public
#endif
#endif
  int type;
  int tagnum;
  int typenum;
#ifndef G__OLDIMPLEMENTATION1259
  G__SIGNEDCHAR_T isconst;
#endif
  struct G__DUMMY_FOR_CINT7 dummyForCint7;
#ifdef __cplusplus
}
#else
} G__value
#endif
;

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

/**************************************************************************
* struct forward declaration; real ones are in common.h
**************************************************************************/
struct G__ifunc_table;
struct G__var_array;
struct G__dictposition;
struct G__comment_info;
struct G__friendtag;
#ifdef G__ASM_WHOLEFUNC
struct G__bytecodefunc;
#endif
struct G__funcentry;
#ifdef G__VMS
struct G__funcentry_VMS;
struct G__ifunc_table_VMS;
#endif
struct G__ifunc_table;
struct G__inheritance;
struct G__var_array;
struct G__tagtable;
struct G__input_file;
#ifdef G__CLINK
struct G__tempobject_list;
#endif
struct G__va_list_para;

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
* struct declaration to avoid error (compiler dependent)
**************************************************************************/
struct G__ifunc_table;
struct G__var_array;
struct G__dictposition; // decl in Api.h because of Cint7's having C++ content

/**************************************************************************
* comment information
*
**************************************************************************/
struct G__comment_info {
  union {
    char  *com;
    fpos_t pos;
  } p;
  int   filenum;
};

/**************************************************************************
* ROOT special requirement
*
**************************************************************************/
#define G__NOSTREAMER      0x01
#define G__NOINPUTOPERATOR 0x02
#define G__USEBYTECOUNT    0x04
#define G__HASVERSION      0x08

struct G__RootSpecial {
  char* deffile;
  int defline;
  char* impfile;
  int impline;
  int version;
  unsigned int instancecount;
  unsigned int heapinstancecount;
  void* defaultconstructor; /* defaultconstructor wrapper/stub pointer */
  struct G__ifunc_table* defaultconstructorifunc; /* defaultconstructor ifunc entry */
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
* structure for function entry
*
**************************************************************************/
#define G__BYTECODE_NOTYET    1
#define G__BYTECODE_FAILURE   2
#define G__BYTECODE_SUCCESS   3
#define G__BYTECODE_ANALYSIS  4 /* ON1164 */


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


#if defined(__cplusplus) && !defined(__CINT__)
}   /* extern C 1 */
#endif

/**************************************************************************
* Interface Method type
*
**************************************************************************/
#if defined(__cplusplus) && defined(G__CPPIF_EXTERNC) && !defined(__CINT__)
extern "C" {   /* extern C 2 */
#endif

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

#if defined(__cplusplus) && defined(G__CPPIF_EXTERNC) && !defined(__CINT__)
} /* extern C 2 */
#endif

#if defined(__cplusplus) && !defined(__CINT__)
extern "C" { /* extern C 3 */
#endif

/**************************************************************************
* structure for class inheritance
*
**************************************************************************/
#define G__ISDIRECTINHERIT         0x0001
#define G__ISVIRTUALBASE           0x0002
#define G__ISINDIRECTVIRTUALBASE   0x0004


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
* make hash value
*
**************************************************************************/

#define G__hash(string,hash,len) len=hash=0;while(string[len]!='\0')hash+=string[len++];


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
* Completion list and function pointer list
*
**************************************************************************/
typedef struct {
  char *name;
  void (*pfunc)();
} G__COMPLETIONLIST;

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
#define G__PVOID ((long)(-1))
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
/**************************************************************************
* Exported Functions
*
**************************************************************************/
#ifdef G__ANSI
#define G__P(funcparam) funcparam
#else
#define G__P(funcparam) ()
#endif

extern G__EXPORT unsigned long G__uint G__P((G__value buf));

#if defined(G__DEBUG) && !defined(G__MEMTEST_C)
#include "src/memtest.h"
#endif

#ifndef G__WILDCARD
#define G__WILDCARD
#endif

extern int G__fgetline G__P((char *string));
extern int G__load G__P((char *commandfile));
/* extern float G__float G__P((G__value buf));*/
extern G__value (*G__GetSpecialObject) G__P((const char *name,void **ptr,void** ppdict));
extern G__EXPORT int  G__call_setup_funcs G__P((void));
extern void G__reset_setup_funcs G__P((void));
extern G__EXPORT const char *G__cint_version G__P((void));
extern void G__init_garbagecollection G__P((void));
extern int G__garbagecollection G__P((void));
extern void G__add_alloctable G__P((void* allocedmem,int type,int tagnum));
extern int G__del_alloctable G__P((void* allocmem));
extern int G__add_refcount G__P((void* allocedmem,void** storedmem));
extern int G__del_refcount G__P((void* allocedmem,void** storedmem));
extern int G__disp_garbagecollection G__P((FILE* fout));
struct G__ifunc_table *G__get_methodhandle G__P((const char *funcname,const char *argtype
                                           ,struct G__ifunc_table *p_ifunc
                                           ,long *pifn,long *poffset
                                           ,int withConversion
                                           ,int withInheritance));
struct G__ifunc_table *G__get_methodhandle2 G__P((char *funcname
                                           ,struct G__param* libp
                                           ,struct G__ifunc_table *p_ifunc
                                           ,long *pifn,long *poffset
                                           ,int withConversion
                                           ,int withInheritance));
struct G__var_array *G__searchvariable G__P((char *varname,int varhash
                                       ,struct G__var_array *varlocal
                                       ,struct G__var_array *varglobal
                                       ,long *pG__struct_offset
                                       ,long *pstore_struct_offset
                                       ,int *pig15
                                       ,int isdecl));


struct G__ifunc_table* G__p2f2funchandle G__P((void* p2f,struct G__ifunc_table* p_ifunc,int* pindex));
G__EXPORT char* G__p2f2funcname G__P((void *p2f));
G__EXPORT int G__isinterpretedp2f G__P((void* p2f));
int G__compile_bytecode G__P((struct G__ifunc_table* ifunc,int index));



/**************************************************************************
 * Variable argument, byte layout policy
 **************************************************************************/
#define G__VAARG_SIZE 1024

typedef struct {
  union {
    char d[G__VAARG_SIZE];
    long i[G__VAARG_SIZE/sizeof(long)];
  } x;
} G__va_arg_buf;


#if (defined(__i386__) && (defined(__linux) || defined(__APPLE__))) || \
    defined(_WIN32) || defined(G__CYGWIN)
/**********************************************
 * Intel architecture, aligns in multiple of 4
 *    |1111|22  |3   |44444444|55555555555555  |
 **********************************************/
#define G__VAARG_INC_COPY_N 4

#elif (defined(__linux)&&defined(__ia64__))
/**********************************************
 * Itanium/linux, aligns in multiple of 8
 **********************************************/

#define G__VAARG_INC_COPY_N 8
#define G__VAARG_PASS_BY_REFERENCE 8

#elif defined(__hpux) || defined(__hppa__)
/**********************************************
 * HP-Precision Architecture,
 *  Args > 8 bytes are passed by reference.  Args > 4 and <= 8 are
 *  right-justified in 8 bytes.  Args <= 4 are right-justified in
 *  4 bytes.
 **********************************************/
/* #define G__VAARG_NOSUPPORT */

#ifdef __ia64__
#define G__VAARG_INC_COPY_N 8
#else
#define G__VAARG_INC_COPY_N 4
#endif
#define G__VAARG_PASS_BY_REFERENCE 8

#elif defined(__x86_64__) && (defined(__linux) || defined(__APPLE__) || \
      defined(__FreeBSD__) && defined(__sun))
/**********************************************
 * AMD64/EM64T
 * It turned out it is quite difficult to support this
 * platform as it uses registers for passing arguments (first 6 long
 * and first 8 double arguments in registers, the remaining on the stack)
 * for Linux/gcc.
 **********************************************/

#define G__VAARG_INC_COPY_N 8
/* #define G__VAARG_PASS_BY_REFERENCE 8 */

#elif defined(__sparc) || defined(__sparc__) || defined(__SUNPRO_C) || \
      defined(__SUNPRO_CC)
/**********************************************
 * Sun Sparc architecture
 * Alignment is similar to Intel, but class/struct
 * objects are passed by reference
 **********************************************/
/* #define G__VAARG_NOSUPPORT */

#define G__VAARG_INC_COPY_N 4
#define G__VAARG_PASS_BY_REFERENCE 8

#elif (defined(__PPC__)||defined(__ppc__))&&(defined(_AIX)||defined(__APPLE__))
/**********************************************
 * PowerPC, AIX and Apple Mac
 * It turned out it is quite difficult if not impossible to support PowerPC.
 * PPC uses registers for passing arguments (general purpose 3-10, floating 1)
 **********************************************/
#if !defined(__GNUC__)
/* Looks like gcc3.3 doesn't use registers. */
#define G__VAARG_NOSUPPORT
#endif
#define G__VAARG_INC_COPY_N 4
#define G__VAARG_PASS_BY_REFERENCE 8

#elif (defined(__PPC__)||defined(__ppc__))&&(defined(__linux)||defined(__linux__))
/**********************************************
 * PowerPC, Linux
 **********************************************/
#define G__VAARG_INC_COPY_N 4
#define G__VAARG_PASS_BY_REFERENCE 8

#elif (defined(__mips__)&&defined(__linux__))
/**********************************************
* MIPS, Linux
* 3 different calling conventions:
*                             | mips   | mipsn32   | mips64
*                             | mipsel | mipsn32el | mips64el
*  ---------------------------+--------+---------------------
*  G__VAARG_INC_COPY_N        |   4    |     8     |    8
*  G__VAARG_PASS_BY_REFERENCE |   4    |     8     |    8
*
* Assuming that
*    G__VAARG_INC_COPY_N
*       is meant to be the size of the argument registers,
*    G__VAARG_PASS_BY_REFERENCE
*       is the number of arguments passed by reference through
*       registers.
*
* Thanks to Thiemo Seufer <ths@networkno.de> of Debian
**********************************************/
# if _MIPS_SIM == _ABIO32 /* mips or mipsel */
#  define G__VAARG_INC_COPY_N 4
#  define G__VAARG_PASS_BY_REFERENCE 4
# elif _MIPS_SIM == _ABIN32 /* mipsn32 or mipsn32el */
#  define G__VAARG_INC_COPY_N 8
#  define G__VAARG_PASS_BY_REFERENCE 8
# elif _MIPS_SIM == _ABI64 /* mips64 or mips64el */
#  define G__VAARG_INC_COPY_N 8
#  define G__VAARG_PASS_BY_REFERENCE 8
# else
#  define G__VAARG_NOSUPPORT
# endif
#else
/**********************************************
 * Other platforms,
 *  Try copying object as value.
 **********************************************/
#define G__VAARG_NOSUPPORT
#define G__VAARG_INC_COPY_N 4
/* #define G__VAARG_PASS_BY_REFERENCE 8 */

#endif


struct G__va_list_para {
  struct G__param *libp;
  int i;
};

void G__va_arg_setalign G__P((int n));
void G__va_arg_copyvalue G__P((int t,void* p,G__value* pval,int objsize));


#endif /* __CINT__ */

/*
#define G__letdouble(&buf,valtype,value) buf->type=valtype;buf->obj.d=value
#define G__letint(&buf,valtype,value)    buf->type=valtype;buf->obj.i=value
*/

#ifndef __CINT__
/*************************************************************************
* ROOT script compiler
*************************************************************************/
extern G__EXPORT void G__Set_RTLD_NOW G__P((void));
extern G__EXPORT void G__Set_RTLD_LAZY G__P((void));
extern G__EXPORT int (*G__ScriptCompiler) G__P((G__CONST char*,G__CONST char*));
extern G__EXPORT void G__RegisterScriptCompiler G__P((int(*p2f)(G__CONST char*,G__CONST char*)));
/*************************************************************************
* Pointer to function evaluation function
*************************************************************************/

/*************************************************************************
* G__atpause, G__aterror API
*************************************************************************/

/*************************************************************************
* interface method setup functions
*************************************************************************/

extern G__EXPORT int G__defined_tagname G__P((G__CONST char* tagname,int noerror));
extern G__EXPORT struct G__Definedtemplateclass *G__defined_templateclass G__P((const char *name));

int G__EXPORT G__deleteglobal G__P((void* p));
int G__EXPORT G__deletevariable G__P((G__CONST char* varname));
extern G__EXPORT int G__optimizemode G__P((int optimizemode));
extern G__EXPORT int G__getoptimizemode G__P((void));
G__value G__string2type_body G__P((G__CONST char *typenamin,int noerror));
G__value G__string2type G__P((G__CONST char *typenamin));
extern G__EXPORT void* G__findsym G__P((G__CONST char *fname));

extern G__EXPORT int G__IsInMacro G__P((void));
extern G__EXPORT void G__storerewindposition G__P((void));
extern G__EXPORT void G__rewinddictionary G__P((void));
extern G__EXPORT void G__SetCriticalSectionEnv G__P(( int(*issamethread)G__P((void)), void(*storelockthread)G__P((void)), void(*entercs)G__P((void)), void(*leavecs)G__P((void)) ));

extern G__EXPORT void G__storelasterror G__P((void));


extern G__EXPORT void G__set_smartunload G__P((int smartunload));

extern G__EXPORT void G__set_autoloading G__P((int (*p2f) G__P((char*))));

extern G__EXPORT void G__set_class_autoloading_callback G__P((int (*p2f) G__P((char*,char*))));
extern G__EXPORT void G__set_class_autoloading_table G__P((char* classname,char* libname));
extern G__EXPORT char* G__get_class_autoloading_table G__P((char* classname));
extern G__EXPORT int G__set_class_autoloading G__P((int newvalue));

typedef int (*G__IgnoreInclude) G__P((const char* fname,const char* expandedfname));

#ifdef G__NEVER
extern G__EXPORT void* G__operator_new G__P((size_t size,void* p));
extern G__EXPORT void* G__operator_new_ary G__P((size_t size,void* p)) ;
extern G__EXPORT void G__operator_delete G__P((void *p)) ;
extern G__EXPORT void G__operator_delete_ary G__P((void *p)) ;
#endif

extern G__EXPORT int G__getexitcode G__P((void));
extern G__EXPORT int G__get_return G__P((int *exitval));

#ifndef G__OLDIMPLEMENTATION1485
#ifdef G__FIX1
extern G__EXPORT int G__fprinterr (FILE* fp,const char* fmt,...);
#else
extern G__EXPORT int G__fprinterr G__P((FILE* fp,const char* fmt,...));
#endif
extern G__EXPORT int G__fputerr G__P((int c));
#else
#define G__fprinterr  fprintf
#endif

extern G__EXPORT void G__SetUseCINTSYSDIR G__P((int UseCINTSYSDIR));
extern G__EXPORT void G__SetCINTSYSDIR G__P((char* cintsysdir));
extern G__EXPORT void G__set_eolcallback G__P((void* eolcallback));
typedef void G__parse_hook_t ();
extern G__EXPORT G__parse_hook_t* G__set_beforeparse_hook G__P((G__parse_hook_t* hook));
extern G__EXPORT void G__set_ioctortype_handler G__P((int (*p2f) G__P((const char*))));
extern G__EXPORT void G__SetCatchException G__P((int mode));
extern G__EXPORT int G__GetCatchException G__P((void));
extern G__EXPORT int G__Lsizeof G__P((const char *typenamein));

#ifdef G__ASM_WHOLEFUNC
/**************************************************************************
* Interface method to run bytecode function
**************************************************************************/
extern G__EXPORT int G__exec_bytecode G__P((G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash));
#endif


/**************************************************************************
 * Exported Cint API functions
 **************************************************************************/
#ifdef G__FIX1
extern G__EXPORT int G__fprintf (FILE* fp,const char* fmt,...);
#else
extern G__EXPORT int G__fprintf G__P((FILE* fp,const char* fmt,...));
#endif
extern G__EXPORT int G__setmasksignal G__P((int));
extern G__EXPORT void G__settemplevel G__P((int val));
extern G__EXPORT void G__clearstack G__P((void));
extern G__EXPORT int G__lasterror G__P((void)) ;
extern G__EXPORT void G__reset_lasterror G__P((void));
extern G__EXPORT int G__gettempfilenum G__P((void));
extern G__EXPORT void G__LockCpp G__P((void));
extern G__EXPORT void G__set_sym_underscore G__P((int x));
extern G__EXPORT int G__get_sym_underscore G__P((void));
extern G__EXPORT void* G__get_errmsgcallback G__P((void));
extern G__EXPORT void G__mask_errmsg G__P((char* msg));

#if (!defined(G__MULTITHREADLIBCINTC)) && (!defined(G__MULTITHREADLIBCINTCPP))

extern G__EXPORT int G__main G__P((int argc,char **argv));
extern G__EXPORT void G__setothermain G__P((int othermain));
extern G__EXPORT void G__exit G__P((int rtn));
extern G__EXPORT int G__getnumbaseclass G__P((int tagnum));
extern G__EXPORT void G__setnewtype G__P((int globalcomp,G__CONST char* comment,int nindex));
extern G__EXPORT void G__setnewtypeindex G__P((int j,int index));
extern G__EXPORT void G__resetplocal G__P((void));
extern G__EXPORT long G__getgvp G__P((void));
extern G__EXPORT void G__resetglobalenv G__P((void));
extern G__EXPORT void G__lastifuncposition G__P((void));
extern G__EXPORT void G__resetifuncposition G__P((void));
extern G__EXPORT void G__setnull G__P((G__value* result));
extern G__EXPORT long G__getstructoffset G__P((void));
extern G__EXPORT int G__getaryconstruct G__P((void));
extern G__EXPORT long G__gettempbufpointer G__P((void));
extern G__EXPORT void G__setsizep2memfunc G__P((int sizep2memfunc));
extern G__EXPORT int G__getsizep2memfunc G__P((void));
extern G__EXPORT int G__get_linked_tagnum G__P((G__linked_taginfo *p));
extern G__EXPORT int G__get_linked_tagnum_fwd G__P((G__linked_taginfo *p));
extern G__EXPORT int G__tagtable_setup G__P((int tagnum,int size,int cpplink,int isabstract,G__CONST char *comment,G__incsetup setup_memvar,G__incsetup setup_memfunc));
extern G__EXPORT int G__search_tagname G__P((G__CONST char *tagname,int type));
extern G__EXPORT int G__search_typename G__P((G__CONST char *typenamein,int typein,int tagnum,int reftype));
extern G__EXPORT int G__defined_typename G__P((G__CONST char* typenamein));
extern G__EXPORT int G__tag_memvar_setup G__P((int tagnum));
extern G__EXPORT int G__memvar_setup G__P((void *p,int type,int reftype,int constvar,int tagnum,int typenum,int statictype,int access,G__CONST char *expr,int definemacro,G__CONST char *comment));
extern G__EXPORT int G__tag_memvar_reset G__P((void));
extern G__EXPORT int G__tag_memfunc_setup G__P((int tagnum));
extern G__EXPORT void G__set_tagnum(G__value* val, int tagnum);
extern G__EXPORT void G__set_typenum(G__value* val, const char* type);
extern G__EXPORT void G__set_type(G__value* val, char* type);

#ifdef G__TRUEP2F
extern G__EXPORT int G__memfunc_setup G__P((G__CONST char *funcname,int hash,G__InterfaceMethod funcp,int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment,void* tp2f,int isvirtual));
#else /* G__TRUEP2F */
extern G__EXPORT int G__memfunc_setup G__P((G__CONST char *funcname,int hash,G__InterfaceMethod funcp,int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment));
#endif /* G__TRUEP2F */

#ifdef G__TRUEP2F
extern G__EXPORT int G__memfunc_setup2 G__P((G__CONST char *funcname,int hash,G__CONST char *mangled_name,G__InterfaceMethod funcp,int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment,void* tp2f,int isvirtual));
#else /* G__TRUEP2F */
extern G__EXPORT int G__memfunc_setup2 G__P((G__CONST char *funcname,int hash,G__CONST char *mangled_name,G__InterfaceMethod funcp,int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment));
#endif /* G__TRUEP2F */

extern G__EXPORT int G__memfunc_next G__P((void));
extern G__EXPORT int G__memfunc_para_setup G__P((int ifn,int type,int tagnum,int typenum,int reftype,G__value *para_default,char *para_def,char *para_name));
extern G__EXPORT int G__tag_memfunc_reset G__P((void));
extern G__EXPORT void G__letint G__P((G__value *buf,int type,long value));
extern G__EXPORT void G__letdouble G__P((G__value *buf,int type,double value));
extern G__EXPORT int G__value_get_type G__P((G__value* buf));
extern G__EXPORT int G__value_get_tagnum G__P((G__value* buf));
extern G__EXPORT void G__store_tempobject G__P((G__value reg));
extern G__EXPORT int G__inheritance_setup G__P((int tagnum,int basetagnum,long baseoffset,int baseaccess,int property));
extern G__EXPORT void G__add_compiledheader G__P((G__CONST char *headerfile));
extern G__EXPORT void G__add_ipath G__P((G__CONST char *ipath));
extern G__EXPORT int G__delete_ipath G__P((G__CONST char *ipath));
extern G__EXPORT struct G__includepath *G__getipathentry();
extern G__EXPORT void G__add_macro G__P((G__CONST char *macro));
extern G__EXPORT void G__check_setup_version G__P((int version,G__CONST char *func));
extern G__EXPORT long G__int G__P((G__value buf));
extern G__EXPORT long G__int_cast G__P((G__value buf));
extern G__EXPORT double G__double G__P((G__value buf));
extern G__EXPORT G__value G__calc G__P((G__CONST char *expr));
extern G__EXPORT int  G__loadfile G__P((G__CONST char* filename));
extern G__EXPORT int  G__setfilecontext G__P((G__CONST char* filename, struct G__input_file* ifile));
extern G__EXPORT int  G__unloadfile G__P((G__CONST char* filename));
extern G__EXPORT int G__init_cint G__P((G__CONST char* command));
extern G__EXPORT void G__scratch_all G__P((void));
extern G__EXPORT void G__setdouble G__P((G__value *pbuf,double d,void* pd,int type,int tagnum,int typenum,int reftype));
extern G__EXPORT void G__setint G__P((G__value *pbuf,long d,void* pd,int type,int tagnum,int typenum,int reftype));
extern G__EXPORT void G__stubstoreenv G__P((struct G__StoreEnv *env,void* p,int tagnum));
extern G__EXPORT void G__stubrestoreenv G__P((struct G__StoreEnv *env));
extern G__EXPORT int G__getstream G__P((const char *source,int *isrc,char *string,const char *endmark));
extern G__EXPORT char *G__type2string G__P((int type,int tagnum,int typenum,int reftype,int isconst));
extern G__EXPORT void G__alloc_tempobject G__P((int tagnum,int typenum));
extern G__EXPORT void G__alloc_tempobject_val G__P((G__value* val));
extern G__EXPORT void G__set_p2fsetup G__P((void (*p2f)()));
extern G__EXPORT void G__free_p2fsetup G__P((void));
extern G__EXPORT int G__genericerror G__P((G__CONST char *message));
extern G__EXPORT char* G__tmpnam G__P((char* name));
extern G__EXPORT int G__setTMPDIR G__P((char* badname));
extern G__EXPORT void G__setPrerun G__P((int prerun));
extern G__EXPORT int G__readline G__P((FILE *fp,char *line,char *argbuf,int *argn,char *arg[]));
extern G__EXPORT int G__getFuncNow G__P((void));
extern G__EXPORT G__value G__getfunction G__P((const char *item,int *known3,int memfunc_flag));
extern FILE* G__getIfileFp G__P((void));
extern G__EXPORT void G__incIfileLineNumber G__P((void));
extern G__EXPORT struct G__input_file* G__get_ifile G__P((void));
extern G__EXPORT void G__setReturn G__P((int rtn));
extern G__EXPORT int G__getPrerun G__P((void));
extern G__EXPORT short G__getDispsource G__P((void));
extern G__EXPORT FILE* G__getSerr G__P((void));
extern G__EXPORT int G__getIsMain G__P((void));
extern G__EXPORT void G__setIsMain G__P((int ismain));
extern G__EXPORT void G__setStep G__P((int step));
extern G__EXPORT int G__getStepTrace G__P((void));
extern G__EXPORT void G__setDebug G__P((int dbg));
extern G__EXPORT int G__getDebugTrace G__P((void));
extern G__EXPORT void G__set_asm_noverflow G__P((int novfl));
extern G__EXPORT int G__get_no_exec G__P((void));
extern G__EXPORT int G__get_no_exec_compile G__P((void));
extern G__EXPORT void G__setdebugcond G__P((void));
extern G__EXPORT int G__init_process_cmd G__P((void));
extern G__EXPORT int G__process_cmd G__P((char *line,char *prompt,int *more,int *err,G__value *rslt));
extern G__EXPORT int G__pause G__P((void));
extern G__EXPORT char* G__input G__P((const char* prompt));
extern G__EXPORT int G__split G__P((char *line,char *string,int *argc,char **argv));
extern G__EXPORT int G__getIfileLineNumber G__P((void));
extern G__EXPORT void G__addpragma G__P((char* comname,void (*p2f) G__P((char*)) ));
extern G__EXPORT void G__add_setup_func G__P((G__CONST char *libname, G__incsetup func));
extern G__EXPORT void G__remove_setup_func G__P((G__CONST char *libname));
extern G__EXPORT void G__setgvp G__P((long gvp));
extern G__EXPORT void G__set_stdio_handle G__P((FILE* sout,FILE* serr,FILE* sin));
extern G__EXPORT void G__setautoconsole G__P((int autoconsole));
extern G__EXPORT int G__AllocConsole G__P((void));
extern G__EXPORT int G__FreeConsole G__P((void));
extern G__EXPORT int G__getcintready G__P((void));
extern G__EXPORT int G__security_recover G__P((FILE* fout));
extern G__EXPORT void G__breakkey G__P((int signame));
extern G__EXPORT int G__stepmode G__P((int stepmode));
extern G__EXPORT int G__tracemode G__P((int tracemode));
extern G__EXPORT int G__setbreakpoint G__P((const char *breakline,const char *breakfile));
extern G__EXPORT int G__getstepmode G__P((void));
extern G__EXPORT int G__gettracemode G__P((void));
extern G__EXPORT int G__printlinenum G__P((void));
extern G__EXPORT int G__search_typename2 G__P((G__CONST char *typenamein,int typein,int tagnum,int reftype,int parent_tagnum));
extern G__EXPORT void G__set_atpause G__P((void (*p2f)()));
extern G__EXPORT void G__set_aterror G__P((void (*p2f)()));
extern G__EXPORT void G__p2f_void_void G__P((void* p2f));
extern G__EXPORT int G__setglobalcomp G__P((int globalcomp));
extern G__EXPORT const char *G__getmakeinfo G__P((const char *item));
extern G__EXPORT const char *G__getmakeinfo1 G__P((const char *item));
extern G__EXPORT int G__get_security_error G__P((void));
extern G__EXPORT char* G__map_cpp_name G__P((const char *in));
extern G__EXPORT char* G__Charref G__P((G__value *buf));
extern G__EXPORT short* G__Shortref G__P((G__value *buf));
extern G__EXPORT int* G__Intref G__P((G__value *buf));
extern G__EXPORT long* G__Longref G__P((G__value *buf));
extern G__EXPORT unsigned char* G__UCharref G__P((G__value *buf));
#ifdef G__BOOL4BYTE
extern G__EXPORT int* G__Boolref G__P((G__value *buf));
#else // G__BOOL4BYTE
extern G__EXPORT unsigned char* G__Boolref G__P((G__value *buf));
#endif // G__BOOL4BYTE
extern G__EXPORT unsigned short* G__UShortref G__P((G__value *buf));
extern G__EXPORT unsigned int* G__UIntref G__P((G__value *buf));
extern G__EXPORT unsigned long* G__ULongref G__P((G__value *buf));
extern G__EXPORT float* G__Floatref G__P((G__value *buf));
extern G__EXPORT double* G__Doubleref G__P((G__value *buf));
extern G__EXPORT int G__loadsystemfile G__P((G__CONST char* filename));
extern G__EXPORT void G__set_ignoreinclude G__P((G__IgnoreInclude ignoreinclude));
extern G__EXPORT G__value G__exec_tempfile_fp G__P((FILE *fp));
extern G__EXPORT G__value G__exec_tempfile G__P((G__CONST char *file));
extern G__EXPORT G__value G__exec_text G__P((G__CONST char *unnamedmacro));
extern G__EXPORT char* G__exec_text_str G__P((G__CONST char *unnamedmacro,char* result));
extern G__EXPORT char* G__lasterror_filename G__P((void));
extern G__EXPORT int G__lasterror_linenum G__P((void));
extern void G__EXPORT G__va_arg_put G__P((G__va_arg_buf* pbuf,struct G__param* libp,int n));

#ifndef G__OLDIMPLEMENTATION1546
extern G__EXPORT const char* G__load_text G__P((G__CONST char *namedmacro));
extern G__EXPORT void G__set_emergencycallback G__P((void (*p2f)()));
#endif
#ifndef G__OLDIMPLEMENTATION1485
extern G__EXPORT void G__set_errmsgcallback(void* p);
#endif
extern G__EXPORT void G__letLonglong G__P((G__value* buf,int type,G__int64 value));
extern G__EXPORT void G__letULonglong G__P((G__value* buf,int type,G__uint64 value));
extern G__EXPORT void G__letLongdouble G__P((G__value* buf,int type,long double value));
extern G__EXPORT G__int64 G__Longlong G__P((G__value buf));
extern G__EXPORT G__uint64 G__ULonglong G__P((G__value buf));
extern G__EXPORT long double G__Longdouble G__P((G__value buf));
extern G__EXPORT G__int64* G__Longlongref G__P((G__value *buf));
extern G__EXPORT G__uint64* G__ULonglongref G__P((G__value *buf));
extern G__EXPORT long double* G__Longdoubleref G__P((G__value *buf));
extern G__EXPORT void G__exec_alloc_lock();
extern G__EXPORT void G__exec_alloc_unlock();

/* Missing interfaces */

extern G__EXPORT int G__clearfilebusy G__P((int));
/* In Api.h: G__InitGetSpecialObject */
/* In Api.h: G__InitUpdateClassInfo */
/* earlier in this file: G__call_setup_funcs */
/* earlier in this file: G__cint_version */
/* earlier in this file: G__clearstack */
extern G__EXPORT void G__CurrentCall(int, void*, long*);
extern G__EXPORT int G__const_resetnoerror G__P((void));
extern G__EXPORT int G__const_setnoerror G__P((void));
extern G__EXPORT int G__const_whatnoerror G__P((void));
extern G__EXPORT void G__enable_wrappers(int set);
extern G__EXPORT int G__wrappers_enabled();
/* earlier in this file: G__deleteglobal */
/* earlier in this file: G__exec_bytecode */
/* earlier in this file: G__findsym */
extern G__EXPORT int G__close_inputfiles G__P((void));
/* earlier in this file: G__p2f2funcname */
extern G__EXPORT void G__scratch_globals_upto G__P((struct G__dictposition *dictpos));
extern G__EXPORT int G__scratch_upto G__P((struct G__dictposition *dictpos));
/* earlier in this file: G__settemplevel */
extern G__EXPORT void G__store_dictposition G__P((struct G__dictposition* dictpos));
extern G__EXPORT int G__printf (const char* fmt,...);
extern G__EXPORT void G__free_tempobject G__P((void));
extern G__EXPORT int G__display_class(FILE *fout,char *name,int base,int start);
extern G__EXPORT int G__display_includepath(FILE *fout);
extern G__EXPORT void G__set_alloclockfunc(void(*)());
extern G__EXPORT void G__set_allocunlockfunc(void(*)());
extern G__EXPORT int G__usermemfunc_setup(char *funcname,int hash,int (*funcp)(),int type,
                         int tagnum,int typenum,int reftype,
                         int para_nu,int ansi,int accessin,int isconst,
                         char *paras, char *comment
#ifdef G__TRUEP2F
                         ,void *truep2f,int isvirtual
#endif
                         ,void *userparam);
extern G__EXPORT char *G__fulltagname(int tagnum,int mask_dollar);
extern G__EXPORT void G__loadlonglong(int* ptag,int* ptype,int which);
extern G__EXPORT int G__isanybase(int basetagnum,int derivedtagnum,long pobject);
extern G__EXPORT int G__pop_tempobject(void);
extern G__EXPORT int G__pop_tempobject_nodel(void);
extern G__EXPORT const char* G__stripfilename(const char* filename);

extern G__EXPORT int G__sizeof (G__value *object);
#ifdef _WIN32
extern G__EXPORT FILE *FOpenAndSleep(const char *filename, const char *mode);
#endif
extern G__EXPORT struct G__ifunc_table_internal *G__get_ifunc_internal(struct G__ifunc_table* iref);

#else /* G__MULTITHREADLIBCINT */

static int (*G__main) G__P((int argc,char **argv));
static void (*G__setothermain) G__P((int othermain));
static int (*G__getnumbaseclass) G__P((int tagnum));
static void (*G__setnewtype) G__P((int globalcomp,G__CONST char* comment,int nindex));
static void (*G__setnewtypeindex) G__P((int j,int index));
static void (*G__resetplocal) G__P((void));
static long (*G__getgvp) G__P((void));
static void (*G__resetglobalenv) G__P((void));
static void (*G__lastifuncposition) G__P((void));
static void (*G__resetifuncposition) G__P((void));
static void (*G__setnull) G__P((G__value* result));
static long (*G__getstructoffset) G__P((void));
static int (*G__getaryconstruct) G__P((void));
static long (*G__gettempbufpointer) G__P((void));
static void (*G__setsizep2memfunc) G__P((int sizep2memfunc));
static int (*G__getsizep2memfunc) G__P((void));
static int (*G__get_linked_tagnum) G__P((G__linked_taginfo *p));
static int (*G__tagtable_setup) G__P((int tagnum,int size,int cpplink,int isabstract,G__CONST char *comment,G__incsetup setup_memvar,G__incsetup setup_memfunc));
static int (*G__search_tagname) G__P((G__CONST char *tagname,int type));
static int (*G__search_typename) G__P((G__CONST char *typenamein,int typein,int tagnum,int reftype));
static int (*G__defined_typename) G__P((G__CONST char* typenamein));
static int (*G__tag_memvar_setup) G__P((int tagnum));
static int (*G__memvar_setup) G__P((void *p,int type,int reftype,int constvar,int tagnum,int typenum,int statictype,int access,G__CONST char *expr,int definemacro,G__CONST char *comment));
static int (*G__tag_memvar_reset) G__P((void));
static int (*G__tag_memfunc_setup) G__P((int tagnum));

#ifdef G__TRUEP2F
static int (*G__memfunc_setup) G__P((G__CONST char *funcname,int hash, G__InterfaceMethod funcp,int type
, int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment,void* tp2f,int isvirtual));
#else /* G__TRUEP2F */
static int (*G__memfunc_setup) G__P((G__CONST char *funcname,int hash, G__InterfaceMethod funcp,int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment));
#endif /* G__TRUEP2F */

#ifdef G__TRUEP2F
static int (*G__memfunc_setup2) G__P((G__CONST char *funcname,int hash, G__CONST char *mangled_name, G__InterfaceMethod funcp
,int type, int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment,void* tp2f,int isvirtual));
#else /* G__TRUEP2F */
static int (*G__memfunc_setup2) G__P((G__CONST char *funcname,int hash,G__CONST char *mangled_name,G__InterfaceMethod funcp,int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment));
#endif /* G__TRUEP2F */

static int (*G__memfunc_next) G__P((void));
static int (*G__memfunc_para_setup) G__P((int ifn,int type,int tagnum,int typenum,int reftype,G__value *para_default,char *para_def,char *para_name));
static int (*G__tag_memfunc_reset) G__P((void));
static void (*G__letint) G__P((G__value *buf,int type,long value));
static void (*G__letdouble) G__P((G__value *buf,int type,double value));
static void (*G__store_tempobject) G__P((G__value reg));
static int (*G__inheritance_setup) G__P((int tagnum,int basetagnum,long baseoffset,int baseaccess,int property));
static void (*G__add_compiledheader) G__P((G__CONST char *headerfile));
static void (*G__add_ipath) G__P((G__CONST char *ipath));
static void (*G__add_macro) G__P((G__CONST char *macro));
static void (*G__check_setup_version) G__P((int version,G__CONST char *func));
static long (*G__int) G__P((G__value buf));
static double (*G__double) G__P((G__value buf));
static G__value (*G__calc) G__P((G__CONST char *expr));
static int  (*G__loadfile) G__P((G__CONST char* filename));
static int  (*G__unloadfile) G__P((G__CONST char* filename));
static int (*G__init_cint) G__P((G__CONST char* command));
static void (*G__scratch_all) G__P((void));
static void (*G__setdouble) G__P((G__value *pbuf,double d,void* pd,int type,int tagnum,int typenum,int reftype));
static void (*G__setint) G__P((G__value *pbuf,long d,void* pd,int type,int tagnum,int typenum,int reftype));
static void (*G__stubstoreenv) G__P((struct G__StoreEnv *env,void* p,int tagnum));
static void (*G__stubrestoreenv) G__P((struct G__StoreEnv *env));
static int (*G__getstream) G__P((char *source,int *isrc,char *string,char *endmark));
static char* (*G__type2string) G__P((int type,int tagnum,int typenum,int reftype,int isconst));
static void (*G__alloc_tempobject) G__P((int tagnum,int typenum));
static void (*G__alloc_tempobject_val) G__P((G__value* val));
static void (*G__set_p2fsetup) G__P((void (*p2f)()));
static void (*G__free_p2fsetup) G__P((void));
static int (*G__genericerror) G__P((G__CONST char *message));
static char* (*G__tmpnam) G__P((char* name));
static int (*G__setTMPDIR) G__P((char* badname));
static void (*G__setPrerun) G__P((int prerun));
static int (*G__readline) G__P((FILE *fp,char *line,char *argbuf,int *argn,char *arg[]));
static int (*G__getFuncNow) G__P((void));
static FILE* (*G__getIfileFp) G__P((void));
static void (*G__incIfileLineNumber) G__P((void));
static struct G__input_file* G__get_ifile G__P((void));
static void (*G__setReturn) G__P((int rtn));
static int (*G__getPrerun) G__P((void));
static short (*G__getDispsource) G__P((void));
static FILE* (*G__getSerr) G__P((void));
static int (*G__getIsMain) G__P((void));
static void (*G__setIsMain) G__P((int ismain));
static void (*G__setStep) G__P((int step));
static int (*G__getStepTrace) G__P((void));
static void (*G__setDebug) G__P((int dbg));
static int (*G__getDebugTrace) G__P((void));
static void (*G__set_asm_noverflow) G__P((int novfl));
static int (*G__get_no_exec) G__P((void));
static int (*G__get_no_exec_compile) G__P((void));
static void (*G__setdebugcond) G__P((void));
static int (*G__init_process_cmd) G__P((void));
static int (*G__process_cmd) G__P((char *line,char *prompt,int *more,int *err,G__value *rslt));
static int (*G__pause) G__P((void));
static char* (*G__input) G__P((const char* prompt));
static int (*G__split) G__P((char *line,char *string,int *argc,char **argv));
static int (*G__getIfileLineNumber) G__P((void));
static void (*G__addpragma) G__P((char* comname,void (*p2f) G__P((char*)) ));
static void (*G__add_setup_func) G__P((G__CONST char *libname,G__incsetup func));
static void (*G__remove_setup_func) G__P((G__CONST char *libname));
static void (*G__setgvp) G__P((long gvp));
static void (*G__set_stdio_handle) G__P((FILE* sout,FILE* serr,FILE* sin));
static void (*G__setautoconsole) G__P((int autoconsole));
static int (*G__AllocConsole) G__P((void));
static int (*G__FreeConsole) G__P((void));
static int (*G__getcintready) G__P((void));
static int (*G__security_recover) G__P((FILE* fout));
static void (*G__breakkey) G__P((int signame));
static int (*G__stepmode) G__P((int stepmode));
static int (*G__tracemode) G__P((int tracemode));
static int (*G__getstepmode) G__P((void));
static int (*G__gettracemode) G__P((void));
static int (*G__printlinenum) G__P((void));
static int (*G__search_typename2) G__P((G__CONST char *typenamein,int typein,int tagnum,int reftype,int parent_tagnum));
static void (*G__set_atpause) G__P((void (*p2f)()));
static void (*G__set_aterror) G__P((void (*p2f)()));
static void (*G__p2f_void_void) G__P((void* p2f));
static int (*G__setglobalcomp) G__P((int globalcomp));
static char* (*G__getmakeinfo) G__P((char *item));
static int (*G__get_security_error) G__P((void));
static char* (*G__map_cpp_name) G__P((char *in));
static char* (*G__Charref) G__P((G__value *buf));
static short* (*G__Shortref) G__P((G__value *buf));
static int* (*G__Intref) G__P((G__value *buf));
static long* (*G__Longref) G__P((G__value *buf));
static unsigned char* (*G__UCharref) G__P((G__value *buf));
static unsigned short* (*G__UShortref) G__P((G__value *buf));
static unsigned int* (*G__UIntref) G__P((G__value *buf));
static unsigned long* (*G__ULongref) G__P((G__value *buf));
static float* (*G__Floatref) G__P((G__value *buf));
static double* (*G__Doubleref) G__P((G__value *buf));
static int (*G__loadsystemfile) G__P((G__CONST char* filename));
static void (*G__set_ignoreinclude) G__P((G__IgnoreInclude ignoreinclude));
static G__value (*G__exec_tempfile) G__P((G__CONST char *file));
static G__value (*G__exec_text) G__P((G__CONST char *unnamedmacro));
static char* (*G__lasterror_filename) G__P((void));
static int (*G__lasterror_linenum) G__P((void));
static void (*G__va_arg_put) G__P((G__va_arg_buf* pbuf,struct G__param* libp,int n));

#ifndef G__OLDIMPLEMENTATION1546
static char* (*G__load_text) G__P((G__CONST char *namedmacro));
static void (*G__set_emergencycallback) G__P((void (*p2f)()));
#endif
#ifndef G__OLDIMPLEMENTATION1485
static void (*G__set_errmsgcallback) G__P((void* p));
#endif
static void (*G__letLonglong) G__P((G__value* buf,int type,G__int64 value));
static void (*G__letULonglong) G__P((G__value* buf,int type,G__uint64 value));
static void (*G__letLongdouble) G__P((G__value* buf,int type,long double value));
static G__int64 (*G__Longlong) G__P((G__value buf)); /* used to be int */
static G__uint64 (*G__ULonglong) G__P((G__value buf)); /* used to be int */
static long double (*G__Longdouble) G__P((G__value buf)); /* used to be int */
static G__int64* (*G__Longlongref) G__P((G__value *buf));
static G__uint64* (*G__ULonglongref) G__P((G__value *buf));
static long double* (*G__Longdoubleref) G__P((G__value *buf));
static struct G__input_file* (*G__get_ifile) G__P((void));

static void (*G__set_alloclockfunc) G__P((void*));
static void (*G__set_allocunlockfunc) G__P((void*));

#ifdef G__MULTITHREADLIBCINTC
G__EXPORT void G__SetCCintApiPointers(
#else
G__EXPORT void G__SetCppCintApiPointers(
#endif
                void* a1,
                void* a2,
                void* a3,
                void* a4,
                void* a5,
                void* a6,
                void* a7,
                void* a8,
                void* a9,
                void* a10,
                void* a11,
                void* a12,
                void* a13,
                void* a14,
                void* a15,
                void* a16,
                void* a17,
                void* a18,
                void* a19,
                void* a20,
                void* a21,
                void* a22,
                void* a23,
                void* a24,
                void* a25,
                void* a26,
                void* a27,
                void* a28,
                void* a29,
                void* a30,
                void* a31,
                void* a32,
                void* a33,
                void* a34,
                void* a35,
                void* a36,
                void* a37,
                void* a38,
                void* a39,
                void* a40,
                void* a41,
                void* a42,
                void* a43,
                void* a44,
                void* a45,
                void* a46,
                void* a47,
                void* a48,
                void* a49,
                void* a50,
                void* a51,
                void* a52,
                void* a53,
                void* a54,
                void* a55,
                void* a56,
                void* a57,
                void* a58,
                void* a59,
                void* a60,
                void* a61,
                void* a62,
                void* a63,
                void* a64,
                void* a65,
                void* a66,
                void* a67,
                void* a68,
                void* a69,
                void* a70,
                void* a71,
                void* a72,
                void* a73,
                void* a74,
                void* a75,
                void* a76,
                void* a77,
                void* a78,
                void* a79,
                void* a80,
                void* a81,
                void* a82,
                void* a83,
                void* a84,
                void* a85,
                void* a86,
                void* a87,
                void* a88,
                void* a89,
                void* a90,
                void* a91,
                void* a92,
                void* a93,
                void* a94,
                void* a95,
                void* a96,
                void* a97,
                void* a100,
                void* a101,
                void* a102,
                void* a103,
                void* a104,
                void* a105,
                void* a106,
                void* a107,
                void* a108,
                void* a109,
                void* a110,
                void* a111,
                void* a112,
                void* a113,
                void* a114,
                void* a115,
                void* a116,
                void* a117,
                void* a118,
                void* a119,
                void* a120,
                void* a121,
                void* a122,
                void* a123,
                void* a124
#ifndef G__OLDIMPLEMENTATION1546
                ,void* a125
                ,void* a126
#endif
#ifndef G__OLDIMPLEMENTATION1485
                ,void* a127
#endif
                ,void* a128
                ,void* a129
                ,void* a130
                ,void* a131
                ,void* a132
                ,void* a133
                ,void* a134
                ,void* a135
                ,void* a136
                ,void* a137
                ,void* a138
                ,void* a139
                ,void* a140
                )
{
  G__main = (int (*) G__P((int argc,char **argv)) ) a1;
  G__setothermain = (void (*) G__P((int othermain)) ) a2;
  G__getnumbaseclass = (int (*) G__P((int tagnum)) ) a3;
  G__setnewtype = (void (*) G__P((int globalcomp,G__CONST char* comment,int nindex)) ) a4;
  G__setnewtypeindex = (void (*) G__P((int j,int index)) ) a5;
  G__resetplocal = (void (*) G__P((void)) ) a6;
  G__getgvp = (long (*) G__P((void)) ) a7;
  G__resetglobalenv = (void (*) G__P((void)) ) a8;
  G__lastifuncposition = (void (*) G__P((void)) ) a9;
  G__resetifuncposition = (void (*) G__P((void)) ) a10;
  G__setnull = (void (*) G__P((G__value* result)) ) a11;
  G__getstructoffset = (long (*) G__P((void)) ) a12;
  G__getaryconstruct = (int (*) G__P((void)) ) a13;
  G__gettempbufpointer = (long (*) G__P((void)) ) a14;
  G__setsizep2memfunc = (void (*) G__P((int sizep2memfunc)) ) a15;
  G__getsizep2memfunc = (int (*) G__P((void)) ) a16;
  G__get_linked_tagnum = (int (*) G__P((G__linked_taginfo *p)) ) a17;
  G__tagtable_setup = (int (*) G__P((int tagnum,int size,int cpplink,int isabstract,G__CONST char *comment,G__incsetup setup_memvar,G__incsetup setup_memfunc)) ) a18;
  G__search_tagname = (int (*) G__P((G__CONST char *tagname,int type)) ) a19;
  G__search_typename = (int (*) G__P((G__CONST char *typenamein,int typein,int tagnum,int reftype)) ) a20;
  G__defined_typename = (int (*) G__P((G__CONST char* typenamein)) ) a21;
  G__tag_memvar_setup = (int (*) G__P((int tagnum)) ) a22;
  G__memvar_setup = (int (*) G__P((void *p,int type,int reftype,int constvar,int tagnum,int typenum,int statictype,int access,G__CONST char *expr,int definemacro,G__CONST char *comment)) ) a23;
  G__tag_memvar_reset = (int (*) G__P((void)) ) a24;
  G__tag_memfunc_setup = (int (*) G__P((int tagnum)) ) a25;

#ifdef G__TRUEP2F
  G__memfunc_setup = (int (*) G__P((G__CONST char *funcname,int hash,G__InterfaceMethod funcp,int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment,void* tp2f,int isvirtual))  ) a26;
#else /* G__TRUEP2F */
  G__memfunc_setup = (int (*) G__P((G__CONST char *funcname,int hash,G__InterfaceMethod funcp,int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment)) )  a26;
#endif /* G__TRUEP2F */

  G__memfunc_next = (int (*) G__P((void)) ) a27;
  G__memfunc_para_setup = (int (*) G__P((int ifn,int type,int tagnum,int typenum,int reftype,G__value *para_default,char *para_def,char *para_name)) ) a28;
  G__tag_memfunc_reset = (int (*) G__P((void)) ) a29;
  G__letint = (void (*) G__P((G__value *buf,int type,long value)) ) a30;
  G__letdouble = (void (*) G__P((G__value *buf,int type,double value)) ) a31;
  G__store_tempobject = (void (*) G__P((G__value reg)) ) a32;
  G__inheritance_setup = (int (*) G__P((int tagnum,int basetagnum,long baseoffset,int baseaccess,int property)) ) a33;
  G__add_compiledheader = (void (*) G__P((G__CONST char *headerfile)) ) a34;
  G__add_ipath = (void (*) G__P((G__CONST char *ipath)) ) a35;
  G__add_macro = (void (*) G__P((G__CONST char *macro)) ) a36;
  G__check_setup_version = (void (*) G__P((int version,G__CONST char *func)) ) a37;
  G__int = (long (*) G__P((G__value buf)) ) a38;
  G__double = (double (*) G__P((G__value buf)) ) a39;
  G__calc = (G__value (*) G__P((G__CONST char *expr)) ) a40;
  G__loadfile = (int (*) G__P((G__CONST char *filename)) ) a41;
  G__unloadfile = (int (*) G__P((G__CONST char *filename)) ) a42;
  G__init_cint = (int (*) G__P((G__CONST char *command)) ) a43;
  G__scratch_all = (void (*) G__P((void)) ) a44;
  G__setdouble = (void (*) G__P((G__value *pbuf,double d,void* pd,int type,int tagnum,int typenum,int reftype)) ) a45;
  G__setint = (void (*) G__P((G__value *pbuf,long d,void* pd,int type,int tagnum,int typenum,int reftype)) ) a46;
  G__stubstoreenv = (void (*) G__P((struct G__StoreEnv *env,void* p,int tagnum)) ) a47;
  G__stubrestoreenv = (void (*) G__P((struct G__StoreEnv *env)) ) a48;
  G__getstream = (int (*) G__P((char *source,int *isrc,char *string,char *endmark)) ) a49;
  G__type2string = (char* (*) G__P((int type,int tagnum,int typenum,int reftype,int isconst)) ) a50;
  G__alloc_tempobject = (void (*) G__P((int tagnum,int typenum)) ) a51;
  G__alloc_tempobject_val = (void (*) G__P((G__value* val)) ) a51;
  G__set_p2fsetup = (void (*) G__P((void (*p2f)())) ) a52;
  G__free_p2fsetup = (void (*) G__P((void)) ) a53;
  G__genericerror = (int (*) G__P((G__CONST char *message)) ) a54;
  G__tmpnam = (char* (*) G__P((char* name)) ) a55;
  G__setTMPDIR = (int (*) G__P((char* badname)) ) a56;
  G__setPrerun = (void (*) G__P((int prerun)) ) a57;
  G__readline = (int (*) G__P((FILE *fp,char *line,char *argbuf,int *argn,char *arg[])) ) a58;
  G__getFuncNow = (int (*) G__P((void)) ) a59;
  G__getIfileFp = (FILE* (*) G__P((void)) ) a60;
  G__incIfileLineNumber = (void (*) G__P((void)) ) a61;
  G__setReturn = (void (*) G__P((int rtn)) ) a62;
  G__getPrerun = (int (*) G__P((void)) ) a63;
  G__getDispsource = (short (*) G__P((void)) ) a64;
  G__getSerr = (FILE* (*) G__P((void)) ) a65;
  G__getIsMain = (int (*) G__P((void)) ) a66;
  G__setIsMain = (void (*) G__P((int ismain)) ) a67;
  G__setStep = (void (*) G__P((int step)) ) a68;
  G__getStepTrace = (int (*) G__P((void)) ) a69;
  G__setDebug = (void (*) G__P((int dbg)) ) a70;
  G__getDebugTrace = (int (*) G__P((void)) ) a71;
  G__set_asm_noverflow = (void (*) G__P((int novfl)) ) a72;
  G__get_no_exec = (int (*) G__P((void)) ) a73;
  G__get_no_exec_compile = (int (*) G__P((void)) ) a74;
  G__setdebugcond = (void (*) G__P((void)) ) a75;
  G__init_process_cmd = (int (*) G__P((void)) ) a76;
  G__process_cmd = (int (*) G__P((char *line,char *prompt,int *more,int *err,G__value *rslt)) ) a77;
  G__pause = (int (*) G__P((void)) ) a78;
  G__input = (char* (*) G__P((const char* prompt)) ) a79;
  G__split = (int (*) G__P((char *line,char *string,int *argc,char **argv)) ) a80;
  G__getIfileLineNumber = (int (*) G__P((void)) ) a81;
  G__addpragma = (void (*) G__P((char* comname,void (*p2f) G__P((char*)) )) ) a82;
  G__add_setup_func = (void (*) G__P((G__CONST char *libname,G__incsetup func)) ) a83;
  G__remove_setup_func = (void (*) G__P((G__CONST char *libname)) ) a84;
  G__setgvp = (void (*) G__P((long gvp)) ) a85;
  G__set_stdio_handle = (void (*) G__P((FILE* sout,FILE* serr,FILE* sin)) ) a86;
  G__setautoconsole = (void (*) G__P((int autoconsole)) ) a87;
  G__AllocConsole = (int (*) G__P((void)) ) a88;
  G__FreeConsole = (int (*) G__P((void)) ) a89;
  G__getcintready = (int (*) G__P((void)) ) a90;
  G__security_recover = (int (*) G__P((FILE* fout)) ) a91;
  G__breakkey = (void (*) G__P((int signame)) ) a92;
  G__stepmode = (int (*) G__P((int stepmode)) ) a93;
  G__tracemode = (int (*) G__P((int tracemode)) ) a94;
  G__getstepmode = (int (*) G__P((void)) ) a95;
  G__gettracemode = (int (*) G__P((void)) ) a96;
  G__printlinenum = (int (*) G__P((void)) ) a97;
  G__search_typename2 = (int (*) G__P((G__CONST char *typenamein,int typein,int tagnum,int reftype,int parent_tagnum)) ) a100;
  G__set_atpause = (void (*) G__P((void (*p2f)())) ) a101;
  G__set_aterror = (void (*) G__P((void (*p2f)())) ) a102;
  G__p2f_void_void = (void (*) G__P((void* p2f)) ) a103;
  G__setglobalcomp = (void (*) G__P((int globalcomp)) ) a104;
  G__getmakeinfo = (char* (*) G__P((char *item)) ) a105;
  G__get_security_error = (int (*) G__P((void)) ) a106;
  G__map_cpp_name = (char* (*) G__P((char *in)) ) a107;
  G__Charref = (char* (*) G__P((G__value *buf)) ) a108;
  G__Shortref = (short* (*) G__P((G__value *buf)) ) a109;
  G__Intref = (int* (*) G__P((G__value *buf)) ) a110;
  G__Longref = (long* (*) G__P((G__value *buf)) ) a111;
  G__UCharref = (unsigned char* (*) G__P((G__value *buf)) ) a112;
  G__UShortref = (unsigned short* (*) G__P((G__value *buf)) ) a113;
  G__UIntref = (unsigned int* (*) G__P((G__value *buf)) ) a114;
  G__ULongref = (unsigned long* (*) G__P((G__value *buf)) ) a115;
  G__Floatref = (float* (*) G__P((G__value *buf)) ) a116;
  G__Doubleref = (double* (*) G__P((G__value *buf)) ) a117;
  G__loadsystemfile = (int (*) G__P((G__CONST char* filename)) ) a118;
  G__set_ignoreinclude = (void (*) G__P((G__IgnoreInclude ignoreinclude)) ) a119;
  G__exec_tempfile = (G__value (*) G__P((G__CONST char *file)) ) a120;
  G__exec_text = (G__value (*) G__P((G__CONST char *unnamedmacro)) ) a121;
  G__lasterror_filename = (char* (*) G__P((void)) ) a122;
  G__lasterror_linenum = (int (*) G__P((void)) ) a123;
  G__va_arg_put = (void (*) G__P((G__va_arg_buf* pbuf,struct G__param* libp,int n)) ) a124;
#ifndef G__OLDIMPLEMENTATION1546
  G__load_text = (char* (*) G__P((G__CONST char *namedmacro)) ) a125;
  G__set_emergencycallback= (void (*) G__P((void (*p2f)())) ) a126;
#endif
#ifndef G__OLDIMPLEMENTATION1485
  G__set_errmsgcallback= (void (*) G__P((void *p)) ) a127;
#endif
  G__letLonglong=(void (*) G__P((G__value* buf,int type,G__int64 value)))a128;
  G__letULonglong=(void (*) G__P((G__value* buf,int type,G__uint64 value)))a129;
  G__letLongdouble=(void (*) G__P((G__value* buf,int type,long double value)))a130;
  G__Longlong=(void (*) G__P((G__value buf)))a131;
  G__ULonglong=(void (*) G__P((G__value buf)))a132;
  G__Longdouble=(void (*) G__P((G__value buf)))a133;
  G__Longlongref=(void (*) G__P((G__value *buf)))a134;
  G__ULonglongref=(void (*) G__P((G__value *buf)))a135;
  G__Longdoubleref=(void (*) G__P((G__value *buf)))a136;
  G__get_ifile = (struct G__intput_ifile* (*) G__P((void)) ) a137;
  G__set_alloclockfunc   = (void (*) G__P((void* foo)) ) a138;
  G__set_allocunlockfunc = (void (*) G__P((void* foo)) ) a139;

  #ifdef G__TRUEP2F
  G__memfunc_setup2 = (int (*) G__P((G__CONST char *funcname,int hash,G__InterfaceMethod funcp,int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment,void* tp2f,int isvirtual))  ) a140;
#else /* G__TRUEP2F */
  G__memfunc_setup2 = (int (*) G__P((G__CONST char *funcname,int hash,G__InterfaceMethod funcp,int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment)) )  a140;
#endif /* G__TRUEP2F */


}

#endif /* G__MULTITHREADLIBCINT */
/**************************************************************************
 * end of Exported Cint API functions
 **************************************************************************/

#endif /* __CINT__ */


#if defined(G__WIN32) && (!defined(G__SYMANTEC)) && defined(G__CINTBODY)
/* ON562 , this used to be test for G__SPECIALSTDIO */
/************************************************************************
* Dummy I/O function for all Win32 based application
************************************************************************/
#ifdef printf
#undef printf
#endif
#ifdef fprintf
#undef fprintf
#endif
#ifdef fputc
#undef fputc
#endif
#ifdef putc
#undef putc
#endif
#ifdef putchar
#undef putchar
#endif
#ifdef fputs
#undef fputs
#endif
#ifdef puts
#undef puts
#endif
#ifdef fgets
#undef fgets
#endif
#ifdef gets
#undef gets
#endif
#ifdef tmpfile
#undef tmpfile
#endif
#define printf  G__printf
#define fprintf G__fprintf
#define fputc   G__fputc
#define putc    G__fputc
#define putchar G__putchar
#define fputs   G__fputs
#define puts    G__puts
#define fgets   G__fgets
#define gets    G__gets
#define system  G__system
#define tmpfile G__tmpfile

#ifdef G__FIX1
int G__printf (const char* fmt,...);
#else
int G__printf G__P((const char* fmt,...));
#endif
int G__fputc G__P((int character,FILE *fp));
int G__putchar G__P((int character));
int G__fputs G__P((char *string,FILE *fp));
int G__puts G__P((char *string));
char *G__fgets G__P((char *string,int n,FILE *fp));
char *G__gets G__P((char *buffer));
int G__system G__P((char *com));
FILE *G__tmpfile(void);

#ifdef G__SPECIALSTDIO

/* THIS IS AN OLD WILDC++ IMPLEMENTATION */
/* signal causes problem in Windows-95 with Tcl/Tk */
#define signal(sigid,p2f)  NULL
#define alarm(time)        NULL

#else /* G__SPECIALSTDIO */

#ifdef signal
#undef signal
#endif
#define signal G__signal
#define alarm(time)        NULL
typedef void (*G__signaltype)(int,void (*)(int));
G__signaltype G__signal G__P((int sgnl,void (*f)(int)));

#endif /* G__SPECIALSTDIO */

#endif /* WIN32 !SYMANTEC CINTBODY*/
/**************************************************************************
* end of specialstdio or win32
**************************************************************************/

/***********************************************************************
 * Native long long support
 ***********************************************************************/
#ifndef __CINT__
extern G__EXPORT G__int64 G__expr_strtoll G__P((const char *nptr,char **endptr, register int base));
extern G__EXPORT G__uint64 G__expr_strtoull G__P((const char *nptr, char **endptr, register int base));
#endif


/***********************************************************************/
#if defined(__cplusplus) && !defined(__CINT__)
} /* extern C 3 */
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
#endif /* __MAKECINT__ */
/**************************************************************************
* endif #ifndef G__MAKECINT
**************************************************************************/



#endif /* G__CI_H */
