/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * CINT header file G__ci.h
 ************************************************************************
 * Description:
 *  C/C++ interpreter parser header file
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation. The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#ifndef G__CI_H
#define G__CI_H

#define G__CINTVERSION 5014038
#define G__CINTVERSIONSTR  "5.14.38, May 5 2000"

/**********************************************************************
* SPECIAL CHANGES and CINT CORE COMPILATION SWITCH
**********************************************************************/

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
 * in platform dependency file OTHMACRO flag.
 */
/* #define G__EXCEPTIONWRAPPER */
/* #define G__STD_EXCEPTION */
#if defined(G__STD_EXCEPTION) && !defined(G__EXCEPTIONWRAPPER)
#define G__EXCEPTIONWRAPPER
#endif

/* If you define G__REFCONV in platform dependency file, bug fix for 
 * reference argument conversion is activated. Reason of not making
 * this default is because it breaks DLL compatibility. If you define
 * G__REFCONV, cint5.14.15 or newer version can load older DLL. But 
 * cint5.14.14 or older version can not load DLL that is created by
 * cint5.14.15 or later. */
#define G__REFCONV

/* This change activates bytecode compilation of class object 
 * instantiation in a function. Because the change includes some
 * problems , it is turned off at this moment by defining following
 * macro. */
#define G__OLDIMPLEMENTATION1073
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


#define G__OLDIMPLEMENTATION834 /* THIS MODIFICATION IS TURNED OFF */


#ifndef G__OLDIMPLEMENTATION1231
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

#endif

/**********************************************************************
* Define following macro if you want to know where global variable is defined. 
* This macro is usually not defined to keep backward compatibility of DLL.
**********************************************************************/
/* #define G__VARIABLEFPOS */
/* #define G__TYPEDEFFPOS */

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

#ifdef _MSC_VAR
#ifndef G__VISUAL
#define G__VISUAL
#endif
#ifndef G__MSC_VAR
#define G__MSC_VAR
#endif
#endif

#ifdef __VMS
#define G__VMS
#endif

#if defined(__BORLANDC__) || defined(__BCPLUSPLUS)
#ifndef G__BORLAND
#define G__BORLAND
#endif
#endif

#if defined(_WIN32) || defined(_WINDOWS) || defined(_Windows) || defined(_WINDOWS_)
#ifndef G__WIN32
#define G__WIN32
#endif
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

#ifdef G__VMS
#ifndef G__NONSCALARFPOS
#define G__NONSCALARFPOS
#endif
typedef long fpos_tt; /* pos_t is defined to be a struct{32,32} in VMS.
                         Therefore,pos_tt is defined to be a long. This
                         is used in G__ifunc_table_VMS, G__functentry_VMS*/
#endif

#ifdef G__BORLAND
#define G__DLLEXPORT __declspec(dllexport)
#else
#define G__DLLEXPORT
#endif

#if defined(G__BORLAND) && defined(G__CINTBODY)
#define G__EXPORT __declspec(dllexport)
#else
#define G__EXPORT
#endif

#if 0
#ifdef G__ROOT
# ifndef __CINT__
#  if defined(WIN32) && defined(_DLL)
#   define DllImport   __declspec(dllimport)
#   define DllExport   __declspec(dllexport)
#  else
#   define DllImport 
#   define DllExport 
#  endif
#  ifdef G__CINTBODY
#   define G__EXTERN   DllExport extern
#  else
#   define G__EXTERN   DllImport extern
#  endif
# endif
G__EXTERN short G__othermain;
G__EXTERN int G__globalcomp;
#endif
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

#ifndef G__OLDIMPLEMENTATION575
#define G__CHECKTYPE(p,t1,t2) \
 if(G__check_type(p,t1,t2,&libp->para[p],result7,funcname)) return(1)
#endif

#ifndef G__OLDIMPLEMENTATION575
#define G__CHECKNONULL(p,t) \
 if(G__check_nonull(p,t,&libp->para[p],result7,funcname)) return(1)
#else
#define G__CHECKNONULL(p) \
 if(G__check_nonull(p,G__int(libp->para[p]),result7,funcname)) return(1)
#endif

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

#ifndef G__OLDIMPLEMENTATION1210
#ifdef __CINT__
typedef int (*G__IgnoreInclude)();
#endif
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
* It would be nice if G__MAXFUNCPARA, G__MAXSTRUCT and G__MAXTYPEDEF
* are improved.
**************************************************************************/
#ifdef G__LONGBUF
#define G__LONGLINE    4096  /* Length of expression */
#define G__ONELINE     4096  /* Length of subexpression,parameter,argument */
#define G__MAXNAME     4096  /* Variable name */
#else
#define G__LONGLINE    1024  /* Length of expression */
#define G__ONELINE      256  /* Length of subexpression,parameter,argument */
#define G__MAXNAME      128  /* Variable name */
#endif
#define G__MAXFILE      500  /* Max interpreted source file */
#define G__MAXFILENAME  256  /* Max interpreted source file name length */
#define G__MAXPARA      100  /* Number of argument for G__main(argc,argv)   */
#define G__MAXARG       100  /* Number of argument for G__init_cint(char *) */
#define G__MAXFUNCPARA   40  /* Function argument */
#ifndef G__OLDIMPLEMENTATION834
#define G__MAXFUNCPARA2  85  /* Function argument */
#endif
#define G__MAXVARDIM     10  /* Array dimention */
#define G__LENPOST       10  /* length of file name extention */
#define G__MAXBASE       30  /* maximum inheritable class */
#define G__TAGNEST       20  /* depth of nested class */

#ifdef G__WIN32
#define G__MAXSTRUCT   4000  /* struct table */
#define G__MAXTYPEDEF  4000  /* typedef table */
#else
#define G__MAXSTRUCT   4000  /* struct table */
#define G__MAXTYPEDEF  4000  /* typedef table */
#endif

/* G__MAXIFUNC and G__MEMDEPTH are not real limit
 * They are depth of one page of function or variable list
 * If the page gets full, more table is allocated. */
#define G__MAXIFUNC 10
#define G__MEMDEPTH 10


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

#define G__LOCAL    0
#ifdef G__MEMBERFUNC
#define G__MEMBER   2
#define G__GLOBAL   4
#define G__NOTHING  6
#else
#define G__GLOBAL   2
#endif

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
typedef struct {
  union {
    double d;
    long    i; /* used to be int */
    struct G__p2p reftype;
#ifndef G__OLDIMPLEMENTATION845
    char ch;
    short sh;
    int in;
    float fl;
    unsigned char uch;
    unsigned short ush;
    unsigned int uin;
    unsigned long ulo;
#endif
  } obj;
  int type;
  int tagnum;
  int typenum;
#ifdef G__REFERENCETYPE2
  long ref;
#endif
#ifndef G__OLDIMPLEMENTATION1259
  G__SIGNEDCHAR_T isconst;
#endif
} G__value ;


#ifndef G__OLDIMPLEMENTATION833
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

#endif


/**************************************************************************
* include file flags
**************************************************************************/
#define G__USERHEADER 1
#define G__SYSHEADER  2


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
* struct declaration to avoid error (compiler dependent)
**************************************************************************/
struct G__ifunc_table;
struct G__var_array;

#ifdef G__FONS_COMMENT
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
#endif

#ifdef G__ROOTSPECIAL
/**************************************************************************
* ROOT special requirement
*
**************************************************************************/
#define G__NOSTREAMER      0x01
#define G__NOINPUTOPERATOR 0x02
#define G__USEBYTECOUNT   0x04

struct G__RootSpecial {
  char* deffile;
  int defline;
  char* impfile;
  int impline;
  int version;
  unsigned int instancecount;
  unsigned int heapinstancecount;
  void* defaultconstructor;
};
#endif

/**************************************************************************
* structure for friend function and class
*
**************************************************************************/
struct G__friendtag {
  short tagnum;
  struct G__friendtag *next;
};

#ifdef G__ASM_WHOLEFUNC
/**************************************************************************
* bytecode compiled interpreted function
*
**************************************************************************/
struct G__bytecodefunc {
  struct G__ifunc_table *ifunc;
  int ifn;
  struct G__var_array *var;
  int varsize;
  G__value *pstack; /* upper part of stack to store numerical constants */
  int stacksize;
  long *pinst;      /* instruction buffer */
  int instsize;
  char *asm_name;   /* name of used ANSI library function */
};
#endif

/**************************************************************************
* structure for function entry
*
**************************************************************************/
#define G__BYTECODE_NOTYET    1
#define G__BYTECODE_FAILURE   2
#define G__BYTECODE_SUCCESS   3
#define G__BYTECODE_ANALYSIS  4 /* ON1164 */

struct G__funcentry {
  /* file position and pointer for restoring start point */
  fpos_t pos; /* Set if interpreted func body defined, unknown otherwise */
  void *p;  /* FILE* for source file or  int (*)() for compiled function
	     * (void*)NULL if no function body */
  int  line_number; /* -1 if no function body or compiled function */
  short filenum;    /* -1 if compiled function, otherwise interpreted func */
#ifdef G__ASM_FUNC
  unsigned short size; /* size (number of lines) of function */
#endif
#ifdef G__TRUEP2F
  void *tp2f;
#endif
#ifdef G__ASM_WHOLEFUNC
  struct G__bytecodefunc *bytecode;
  int bytecodestatus;
#endif
};

#ifdef G__VMS
/***************************************************************************
*  Need for struct G__ifunc_table_VMS.  Neccessary for
*  Cint_Method::FilePosition().
***************************************************************************/
struct G__funcentry_VMS {
  /* file position and pointer for restoring start point */
  fpos_tt pos; /* Set if interpreted func body defined, unknown otherwise */
  void *p;     /* FILE* for source file or  int (*)() for compiled function
                * (void*)NULL if no function body */
  int  line_number; /* -1 if no function body or compiled function */
  short filenum;    /* -1 if compiled function, otherwise interpreted func */
};
#endif

#ifdef G__OLDIMPLEMENTATION834_YET
/**************************************************************************
* Supporting unlimited number of function arguments
**************************************************************************/
struct G__more_funcarg {
  char para_reftype[G__MAXFUNCPARA];
  char para_type[G__MAXFUNCPARA];
  char para_isconst[G__MAXFUNCPARA];
  short para_p_tagtable[G__MAXFUNCPARA];
  short para_p_typetable[G__MAXFUNCPARA];
  G__value *para_default[G__MAXFUNCPARA];
  char *para_name[G__MAXFUNCPARA];
  char *para_def[G__MAXFUNCPARA];
  struct G__more_funcarg *next;
};
#endif

/**************************************************************************
* structure for ifunc (Interpleted FUNCtion) table
*
**************************************************************************/
struct G__ifunc_table {
  /* number of interpreted function */
  int allifunc;

  /* function name and hash for identification */
  char funcname[G__MAXIFUNC][G__MAXNAME];
  int  hash[G__MAXIFUNC];

  struct G__funcentry entry[G__MAXIFUNC],*pentry[G__MAXIFUNC];

  /* type of return value */
  G__SIGNEDCHAR_T type[G__MAXIFUNC];
  short p_tagtable[G__MAXIFUNC];
  short p_typetable[G__MAXIFUNC];
  G__SIGNEDCHAR_T reftype[G__MAXIFUNC];
  short para_nu[G__MAXIFUNC];
  G__SIGNEDCHAR_T isconst[G__MAXIFUNC];
#ifndef G__OLDIMPLEMENTATION1250
  G__SIGNEDCHAR_T isexplicit[G__MAXIFUNC];
#endif

  /* number and type of function parameter */
  /* G__inheritclass() depends on type of following members */
#ifndef G__OLDIMPLEMENTATION834
  char para_reftype[G__MAXIFUNC][G__MAXFUNCPARA2];
  char para_type[G__MAXIFUNC][G__MAXFUNCPARA2];
  char para_isconst[G__MAXIFUNC][G__MAXFUNCPARA2];
  short para_p_tagtable[G__MAXIFUNC][G__MAXFUNCPARA2];
  short para_p_typetable[G__MAXIFUNC][G__MAXFUNCPARA2];
  G__value *para_default[G__MAXIFUNC][G__MAXFUNCPARA2];
  char *para_name[G__MAXIFUNC][G__MAXFUNCPARA2];
  char *para_def[G__MAXIFUNC][G__MAXFUNCPARA2];
#else
  char para_reftype[G__MAXIFUNC][G__MAXFUNCPARA];
  char para_type[G__MAXIFUNC][G__MAXFUNCPARA];
  char para_isconst[G__MAXIFUNC][G__MAXFUNCPARA];
  short para_p_tagtable[G__MAXIFUNC][G__MAXFUNCPARA];
  short para_p_typetable[G__MAXIFUNC][G__MAXFUNCPARA];
  G__value *para_default[G__MAXIFUNC][G__MAXFUNCPARA];
  char *para_name[G__MAXIFUNC][G__MAXFUNCPARA];
  char *para_def[G__MAXIFUNC][G__MAXFUNCPARA];
#endif

  /* C or C++ */
  char iscpp[G__MAXIFUNC];

  /* ANSI or standard header format */
  char ansi[G__MAXIFUNC];

  /**************************************************
   * if function is called, busy[] is incremented
   **************************************************/
  short busy[G__MAXIFUNC];

  struct G__ifunc_table *next;
  short page;

  G__SIGNEDCHAR_T access[G__MAXIFUNC];  /* private, protected, public */
  char staticalloc[G__MAXIFUNC];

  int tagnum;
  char isvirtual[G__MAXIFUNC]; /* virtual function flag */
  char ispurevirtual[G__MAXIFUNC]; /* virtual function flag */

#ifdef G__FRIEND
  struct G__friendtag *friendtag[G__MAXIFUNC];
#endif

  G__SIGNEDCHAR_T globalcomp[G__MAXIFUNC];

#ifdef G__FONS_COMMENT
  struct G__comment_info comment[G__MAXIFUNC];
#endif

#ifdef G__OLDIMPLEMENTATION834_YET
  struct G__more_funcarg *more_para[G__MAXIFUNC];
#endif
};


#ifdef G__VMS
/**************************************************************************
* For VMS:
*  This is the same struct as G__ifunc_table excep pentry becomes
*  G__funcentry_VMS.  This is needed in Cint_method::FilePosition().
**************************************************************************/
struct G__ifunc_table_VMS {
  /* number of interpreted function */
  int allifunc;

  /* function name and hash for identification */
  char funcname[G__MAXIFUNC][G__MAXNAME];
  int  hash[G__MAXIFUNC];

  struct G__funcentry entry[G__MAXIFUNC];
  struct G__funcentry_VMS *pentry[G__MAXIFUNC];

  /* type of return value */
  G__SIGNEDCHAR_T type[G__MAXIFUNC];
  short p_tagtable[G__MAXIFUNC];
  short p_typetable[G__MAXIFUNC];
  G__SIGNEDCHAR_T reftype[G__MAXIFUNC];
  short para_nu[G__MAXIFUNC];
  G__SIGNEDCHAR_T isconst[G__MAXIFUNC];
#ifndef G__OLDIMPLEMENTATION1250
  G__SIGNEDCHAR_T isexplicit[G__MAXIFUNC];
#endif

  /* number and type of function parameter */
  /* G__inheritclass() depends on type of following members */
#ifndef G__OLDIMPLEMENTATION834
  char para_reftype[G__MAXIFUNC][G__MAXFUNCPARA2];
  char para_type[G__MAXIFUNC][G__MAXFUNCPARA2];
  char para_isconst[G__MAXIFUNC][G__MAXFUNCPARA2];
  short para_p_tagtable[G__MAXIFUNC][G__MAXFUNCPARA2];
  short para_p_typetable[G__MAXIFUNC][G__MAXFUNCPARA2];
  G__value *para_default[G__MAXIFUNC][G__MAXFUNCPARA2];
  char *para_name[G__MAXIFUNC][G__MAXFUNCPARA2];
  char *para_def[G__MAXIFUNC][G__MAXFUNCPARA2];
#else
  char para_reftype[G__MAXIFUNC][G__MAXFUNCPARA];
  char para_type[G__MAXIFUNC][G__MAXFUNCPARA];
  char para_isconst[G__MAXIFUNC][G__MAXFUNCPARA];
  short para_p_tagtable[G__MAXIFUNC][G__MAXFUNCPARA];
  short para_p_typetable[G__MAXIFUNC][G__MAXFUNCPARA];
  G__value *para_default[G__MAXIFUNC][G__MAXFUNCPARA];
  char *para_name[G__MAXIFUNC][G__MAXFUNCPARA];
  char *para_def[G__MAXIFUNC][G__MAXFUNCPARA];
#endif

  /* C or C++ */
  char iscpp[G__MAXIFUNC];

  /* ANSI or standard header format */
  char ansi[G__MAXIFUNC];

  /**************************************************
   * if function is called, busy[] is incremented
   **************************************************/
  short busy[G__MAXIFUNC];

  struct G__ifunc_table *next;
  short page;

  G__SIGNEDCHAR_T access[G__MAXIFUNC];  /* private, protected, public */
  char staticalloc[G__MAXIFUNC];

  int tagnum;
  char isvirtual[G__MAXIFUNC]; /* virtual function flag */
  char ispurevirtual[G__MAXIFUNC]; /* virtual function flag */

#ifdef G__FRIEND
  struct G__friendtag *friendtag[G__MAXIFUNC];
#endif

  G__SIGNEDCHAR_T globalcomp[G__MAXIFUNC];

#ifdef G__FONS_COMMENT
  struct G__comment_info comment[G__MAXIFUNC];
#endif

#ifdef G__OLDIMPLEMENTATION834_YET
  struct G__more_funcarg *more_para[G__MAXIFUNC];
#endif
};
#endif

/**************************************************************************
* structure for function and array parameter
*
**************************************************************************/
struct G__param {
  int paran;
  char parameter[G__MAXFUNCPARA][G__ONELINE];
  G__value para[G__MAXFUNCPARA];
#ifndef G__OLDIMPLEMENTATION834
  int allparan;
  struct G__param *next;
#endif
};


#ifndef G__OLDIMPLEMENTATION1231
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

#ifdef G__ANSIHEADER
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
#endif /* 1231 */

/**************************************************************************
* structure for class inheritance
*
**************************************************************************/
#define G__ISDIRECTINHERIT 0x0001
#define G__ISVIRTUALBASE   0x0002
struct G__inheritance {
  int basen;
  short basetagnum[G__MAXBASE];
#ifdef G__VIRTUALBASE
  long baseoffset[G__MAXBASE];
#else
  int baseoffset[G__MAXBASE];
#endif
  G__SIGNEDCHAR_T baseaccess[G__MAXBASE];
  char property[G__MAXBASE];
};

/**************************************************************************
* structure for variable table
*
**************************************************************************/
struct G__var_array {
  /* union for variable pointer */
  long p[G__MEMDEPTH]; /* used to be int */
  int allvar;
  char varnamebuf[G__MEMDEPTH][G__MAXNAME]; /* variable name */
  int hash[G__MEMDEPTH];                    /* hash table of varname */
  int varlabel[G__MEMDEPTH+1][G__MAXVARDIM];  /* points varpointer */
  short paran[G__MEMDEPTH];
  char bitfield[G__MEMDEPTH];
#ifdef G__VARIABLEFPOS
  int filenum[G__MEMDEPTH];
  int linenum[G__MEMDEPTH];
#endif

  /* type information,
     if pointer : Char,Int,Short,Long,Double,U(struct,union)
     if value   : char,int,short,long,double,u(struct,union) */
  G__SIGNEDCHAR_T  type[G__MEMDEPTH];
  G__SIGNEDCHAR_T constvar[G__MEMDEPTH];
  short p_tagtable[G__MEMDEPTH];        /* tagname if struct,union */
  short p_typetable[G__MEMDEPTH];       /* typename if typedef */
  short statictype[G__MEMDEPTH];
  G__SIGNEDCHAR_T reftype[G__MEMDEPTH];

  /* chain for next G__var_array */
  struct G__var_array *next;

  G__SIGNEDCHAR_T access[G__MEMDEPTH];  /* private, protected, public */

#ifdef G__SHOWSTACK /* not activated */
  struct G__ifunc_table *ifunc;
  int ifn;
  struct G__var_array *prev_local;
  int prev_filenum;
  short prev_line_number;
  long struct_offset;
  int tagnum;
  int exec_memberfunc;
#endif
#ifdef G__VAARG
  struct G__param *libp;
#endif

#ifndef G__NEWINHERIT
  char isinherit[G__MEMDEPTH];
#endif
  G__SIGNEDCHAR_T globalcomp[G__MEMDEPTH];

#ifdef G__FONS_COMMENT
  struct G__comment_info comment[G__MEMDEPTH];
#endif
} ;


/**************************************************************************
* structure struct,union tag information
*
**************************************************************************/


struct G__tagtable {
  /* tag entry information */
  char type[G__MAXSTRUCT]; /* struct,union,enum,class */

  char *name[G__MAXSTRUCT];
  int  hash[G__MAXSTRUCT];
  int  size[G__MAXSTRUCT];
  /* member information */
  struct G__var_array *memvar[G__MAXSTRUCT];
  struct G__ifunc_table *memfunc[G__MAXSTRUCT];
  struct G__inheritance *baseclass[G__MAXSTRUCT];
  int virtual_offset[G__MAXSTRUCT];
  G__SIGNEDCHAR_T globalcomp[G__MAXSTRUCT];
  G__SIGNEDCHAR_T iscpplink[G__MAXSTRUCT];
  char isabstract[G__MAXSTRUCT];

  int  line_number[G__MAXSTRUCT];
  short filenum[G__MAXSTRUCT];

  short parent_tagnum[G__MAXSTRUCT];
  char funcs[G__MAXSTRUCT];
  char istypedefed[G__MAXSTRUCT];
  char istrace[G__MAXSTRUCT];
  char isbreak[G__MAXSTRUCT];
  int  alltag;

#ifdef G__FRIEND
  struct G__friendtag *friendtag[G__MAXSTRUCT];
#endif

#ifdef G__FONS_COMMENT
  struct G__comment_info comment[G__MAXSTRUCT];
#endif

  G__incsetup incsetup_memvar[G__MAXSTRUCT];
  G__incsetup incsetup_memfunc[G__MAXSTRUCT];

#ifdef G__ROOTSPECIAL
  char rootflag[G__MAXSTRUCT];
  struct G__RootSpecial *rootspecial[G__MAXSTRUCT];
#endif

#ifndef G__OLDIMPLEMENTATION1238
  char isctor[G__MAXSTRUCT];
#endif
};

/**************************************************************************
* structure typedef information
*
**************************************************************************/

struct G__typedef {
  char type[G__MAXTYPEDEF];
  char *name[G__MAXTYPEDEF];
  int  hash[G__MAXTYPEDEF];
  short  tagnum[G__MAXTYPEDEF];
  char reftype[G__MAXTYPEDEF];
#ifdef G__CPPLINK1
  char globalcomp[G__MAXTYPEDEF];
#endif
  int nindex[G__MAXTYPEDEF];
  int *index[G__MAXTYPEDEF];
  short parent_tagnum[G__MAXTYPEDEF];
  char iscpplink[G__MAXTYPEDEF];
#ifdef G__FONS_COMMENT
  struct G__comment_info comment[G__MAXSTRUCT];
#endif
#ifdef G__TYPEDEFFPOS
  int filenum[G__MAXTYPEDEF];
  int linenum[G__MAXTYPEDEF];
#endif
  int alltype;
};


/**************************************************************************
* structure for input file
*
**************************************************************************/
struct G__input_file {
  FILE *fp;
  int line_number;
  short filenum;
  char name[G__MAXFILENAME];
};


/**************************************************************************
* tempobject list
*
**************************************************************************/
#ifdef G__CLINK
struct G__tempobject_list {
  G__value obj;
  int  level;
#ifdef G__CPPLINK3
  int cpplink;
#endif
  struct G__tempobject_list *prev;
};
#endif



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
#define G__PVOID (-1)
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

#if defined(G__DEBUG) && !defined(G__MEMTEST_C)
#include "src/memtest.h"
#endif

#ifndef G__WILDCARD
#define G__WILDCARD
#endif

#ifndef G__OLDIMPLEMENTATION1169
extern G__EXPORT char* G__Charref G__P((G__value *buf));
extern G__EXPORT short* G__Shortref G__P((G__value *buf));
extern G__EXPORT int* G__Intref G__P((G__value *buf));
extern G__EXPORT long* G__Longref G__P((G__value *buf));
extern G__EXPORT unsigned char* G__UCharref G__P((G__value *buf));
extern G__EXPORT unsigned int* G__UIntref G__P((G__value *buf));
extern G__EXPORT unsigned short* G__UShortref G__P((G__value *buf));
extern G__EXPORT unsigned long* G__ULongref G__P((G__value *buf));
extern G__EXPORT float* G__Floatref G__P((G__value *buf));
extern G__EXPORT double* G__Doubleref G__P((G__value *buf));
#endif
extern G__EXPORT int G__get_security_error G__P((void));
extern G__EXPORT char* G__map_cpp_name G__P((char *in));
extern G__EXPORT int G__genericerror G__P((char *message));
extern G__EXPORT char* G__tmpnam G__P((char* name));
extern G__EXPORT int G__setTMPDIR G__P((char* badname));
extern G__EXPORT void G__setPrerun G__P((int prerun));
extern G__EXPORT int G__getFuncNow G__P((void));
extern FILE* G__getIfileFp G__P((void));
extern G__EXPORT void G__incIfileLineNumber G__P((void));
extern G__EXPORT int G__getIfileLineNumber G__P((void));
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
extern G__EXPORT int G__process_cmd G__P((char *line,char *prompt,int *more));
extern G__EXPORT void G__setothermain G__P((int othermain));
extern G__EXPORT void G__setglobalcomp G__P((int globalcomp));
extern G__EXPORT int G__main G__P((int argc,char **argv));
extern G__EXPORT int G__readline G__P((FILE *fp,char *line,char *argbuf,int *argn,char *arg[]));
extern int G__fgetline G__P((char *string));
extern G__EXPORT int G__split G__P((char *line,char *string,int *argc,char **argv));
extern G__EXPORT void G__addpragma G__P((char* comname,void (*p2f) G__P((char*)) ));
extern G__EXPORT G__value G__calc G__P((char *expr));
extern G__EXPORT void G__scratch_all G__P((void));
extern G__EXPORT int G__init_cint G__P((char *command));
extern G__EXPORT void G__set_stdio_handle G__P((FILE* sout,FILE* serr,FILE* sin));
extern int G__load G__P((char *commandfile));
extern G__EXPORT char* G__input G__P((char* prompt));
extern G__EXPORT int G__pause G__P((void));
extern G__EXPORT double G__double G__P((G__value buf));
/* extern float G__float G__P((G__value buf));*/
extern G__EXPORT long G__int G__P((G__value buf));
extern G__EXPORT int  G__loadfile G__P((char *filename));
extern G__EXPORT int  G__unloadfile G__P((char *filename));
extern G__EXPORT char *G__getmakeinfo G__P((char *item));
#if !defined(G__OLDIMPLEMENTATION481)
extern G__value (*G__GetSpecialObject) G__P((char *name,void **ptr,void** ppdict));
#elif !defined(G__OLDIMPLEMENTATION455)
extern G__value (*G__GetSpecialObject) G__P((char *name,void *ptr));
#else
extern G__value (*G__GetSpecialObject) G__P((char *name));
#endif
void G__security_recover G__P((FILE *fout));
extern int G__process_cmd G__P((char *line, char *prompt, int *more));
extern G__EXPORT void G__add_setup_func G__P((char *libname, G__incsetup func));
extern G__EXPORT void G__remove_setup_func G__P((char *libname));
extern int  G__call_setup_funcs G__P((void));
extern void G__reset_setup_funcs G__P((void));
extern char *G__cint_version G__P((void));
extern void G__init_garbagecollection G__P((void));
extern int G__garbagecollection G__P((void));
extern void G__add_alloctable G__P((void* allocedmem,int type,int tagnum));
extern int G__del_alloctable G__P((void* allocmem));
extern int G__add_refcount G__P((void* allocedmem,void** storedmem));
extern int G__del_refcount G__P((void* allocedmem,void** storedmem));
extern int G__disp_garbagecollection G__P((FILE* fout));
struct G__ifunc_table *G__get_methodhandle G__P((char *funcname,char *argtype
					   ,struct G__ifunc_table *p_ifunc
					   ,long *pifn,long *poffset));
struct G__var_array *G__searchvariable G__P((char *varname,int varhash
				       ,struct G__var_array *varlocal
				       ,struct G__var_array *varglobal
				       ,long *pG__struct_offset
				       ,long *pstore_struct_offset
				       ,int *pig15
				       ,int isdecl));
extern G__EXPORT void G__setdouble G__P((G__value *pbuf,double d,void* pd,int type,int tagnum,int typenum,int reftype));
extern G__EXPORT void G__setint G__P((G__value *pbuf,long d,void* pd,int type,int tagnum,int typenum,int reftype));
extern G__EXPORT void G__stubstoreenv G__P((struct G__StoreEnv *env,void* p,int tagnum));
extern G__EXPORT void G__stubrestoreenv G__P((struct G__StoreEnv *env));
extern G__EXPORT void G__alloc_tempobject G__P((int tagnum,int typenum));

extern G__EXPORT int G__getstream G__P((char *source,int *isrc,char *string,char *endmark));
extern G__EXPORT char *G__type2string G__P((int type,int tagnum,int typenum,int reftype
			   ,int isconst));

struct G__ifunc_table* G__p2f2funchandle G__P((void* p2f,struct G__ifunc_table* p_ifunc,int* pindex));
char* G__p2f2funcname G__P((void *p2f));
int G__isinterpretedp2f G__P((void* p2f));
int G__compile_bytecode G__P((struct G__ifunc_table* ifunc,int index));

extern G__EXPORT void G__set_p2fsetup G__P((void (*p2f)()));
extern G__EXPORT void G__free_p2fsetup G__P((void));
extern G__EXPORT int G__printlinenum G__P((void));

#endif /* __CINT__ */

/*
#define G__letdouble(&buf,valtype,value) buf->type=valtype;buf->obj.d=value
#define G__letint(&buf,valtype,value)    buf->type=valtype;buf->obj.i=value
*/

#ifndef __CINT__
/*************************************************************************
* ROOT script compiler
*************************************************************************/
#ifndef G__PHILIPPE1
extern G__EXPORT void G__Set_RTLD_NOW G__P((void));
extern G__EXPORT void G__Set_RTLD_LAZY G__P((void));
extern G__EXPORT int (*G__ScriptCompiler) G__P((G__CONST char*,G__CONST char*));
extern G__EXPORT void G__RegisterScriptCompiler G__P((int(*p2f)(G__CONST char*,G__CONST char*)));
#endif
/*************************************************************************
* Pointer to function evaluation function
*************************************************************************/
extern G__EXPORT void G__p2f_void_void G__P((void* p2f));

/*************************************************************************
* G__atpause, G__aterror API
*************************************************************************/
extern G__EXPORT void G__set_atpause G__P((void (*p2f)()));
extern G__EXPORT void G__set_aterror G__P((void (*p2f)()));

/*************************************************************************
* interface method setup functions
*************************************************************************/
extern G__EXPORT int G__getnumbaseclass G__P((int tagnum));
extern G__EXPORT void G__setnewtype G__P((int globalcomp,G__CONST char* comment,int nindex));
extern G__EXPORT void G__setnewtypeindex G__P((int j,int index));
extern G__EXPORT void G__resetplocal G__P(());
extern G__EXPORT long G__getgvp G__P(());
extern G__EXPORT void G__setgvp G__P((long gvp));
extern G__EXPORT void G__resetglobalenv G__P(());
extern G__EXPORT void G__lastifuncposition G__P(());
extern G__EXPORT void G__resetifuncposition G__P(());
extern G__EXPORT void G__setnull G__P((G__value* result));
extern G__EXPORT long G__getstructoffset G__P(());
extern G__EXPORT int G__getaryconstruct G__P(());
extern G__EXPORT long G__gettempbufpointer G__P(());
extern G__EXPORT void G__setsizep2memfunc G__P((int sizep2memfunc));
extern G__EXPORT int G__getsizep2memfunc G__P(());

extern G__EXPORT int G__get_linked_tagnum G__P((G__linked_taginfo *p));
extern G__EXPORT int G__tagtable_setup G__P((int tagnum,int size,int cpplink,int isabstract,G__CONST char *comment,G__incsetup setup_memvar,G__incsetup setup_memfunc));
extern G__EXPORT int G__search_tagname G__P((G__CONST char *tagname,int type));
extern G__EXPORT int G__search_typename G__P((G__CONST char *typenamein,int typein,int tagnum,int reftype));
extern G__EXPORT int G__search_typename2 G__P((G__CONST char *typenamein,int typein,int tagnum,int reftype,int parent_tagnum));
extern int G__defined_tagname G__P((G__CONST char* tagname,int noerror));
extern G__EXPORT int G__defined_typename G__P((G__CONST char* typenamein));
extern G__EXPORT int G__tag_memvar_setup G__P((int tagnum));
extern G__EXPORT int G__memvar_setup G__P((void *p,int type,int reftype,int constvar,int tagnum,int typenum,int statictype,int access,G__CONST char *expr,int definemacro,G__CONST char *comment));
extern G__EXPORT int G__tag_memvar_reset G__P(());
extern G__EXPORT int G__tag_memfunc_setup G__P((int tagnum));

#ifndef G__OLDIMPLEMENTATION1231
#ifdef G__TRUEP2F
extern G__EXPORT int G__memfunc_setup G__P((G__CONST char *funcname,int hash,G__InterfaceMethod funcp,int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment,void* tp2f,int isvirtual));
#else /* G__TRUEP2F */
extern G__EXPORT int G__memfunc_setup G__P((G__CONST char *funcname,int hash,G__InterfaceMethod funcp,int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment));
#endif /* G__TRUEP2F */
#else /* 1231 */
#ifdef G__TRUEP2F
extern G__EXPORT int G__memfunc_setup G__P((G__CONST char *funcname,int hash,int (*funcp)(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash),int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment,void* tp2f,int isvirtual));
#else /* G__TRUEP2F */
extern G__EXPORT int G__memfunc_setup G__P((G__CONST char *funcname,int hash,int (*funcp)(G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash),int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int access,int isconst,G__CONST char *paras,G__CONST char *comment));
#endif /* G__TRUEP2F */
#endif /* 1231 */

extern G__EXPORT int G__memfunc_next G__P(());
extern G__EXPORT int G__memfunc_para_setup G__P((int ifn,int type,int tagnum,int typenum,int reftype,G__value *para_default,char *para_def,char *para_name));
extern G__EXPORT int G__tag_memfunc_reset G__P(());
extern G__EXPORT void G__letint G__P((G__value *buf,int type,long value));
extern G__EXPORT void G__letdouble G__P((G__value *buf,int type,double value));
extern G__EXPORT void G__store_tempobject G__P((G__value reg));
extern G__EXPORT int G__inheritance_setup G__P((int tagnum,int basetagnum,long baseoffset,int baseaccess,int property));
extern G__EXPORT void G__add_compiledheader G__P((G__CONST char *headerfile));
extern G__EXPORT void G__add_ipath G__P((G__CONST char *ipath));
extern G__EXPORT void G__add_macro G__P((G__CONST char *macro));
extern G__EXPORT void G__check_setup_version G__P((int version,G__CONST char *func));
int G__deleteglobal G__P((void* p));
int G__deletevariable G__P((G__CONST char* varname));
extern G__EXPORT void G__setautoconsole G__P((int autoconsole));
extern G__EXPORT int G__AllocConsole G__P(());
extern G__EXPORT int G__FreeConsole G__P(());
extern G__EXPORT int G__getcintready G__P(());
extern G__EXPORT void G__security_recover G__P((FILE* fout));
extern G__EXPORT void G__breakkey G__P((int signame));
extern G__EXPORT int G__tracemode G__P((int tracemode));
extern G__EXPORT int G__stepmode G__P((int stepmode));
extern G__EXPORT int G__gettracemode G__P(());
extern G__EXPORT int G__getstepmode G__P(());
#ifndef G__OLDIMPLEMENTATION1142
extern G__EXPORT int G__optimizemode G__P((int optimizemode));
extern G__EXPORT int G__getoptimizemode G__P(());
#endif
G__value G__string2type_body G__P((G__CONST char *typenamin,int noerror));
G__value G__string2type G__P((G__CONST char *typenamin));
void* G__findsym G__P((G__CONST char *fname));

extern G__EXPORT int G__IsInMacro G__P(());
extern G__EXPORT void G__storerewindposition G__P(());
extern G__EXPORT void G__rewinddictionary G__P(());
extern G__EXPORT void G__SetCriticalSectionEnv G__P(());

#ifndef G__OLDIMPLEMENTATION1198
extern G__EXPORT void G__storelasterror G__P(());
extern G__EXPORT char* G__lasterror_filename G__P(());
extern G__EXPORT int G__lasterror_linenum G__P(());
#endif

#ifndef G__OLDIMPLEMENTATION1207
extern G__EXPORT int G__loadsystemfile G__P((G__CONST char* filename));
#endif

#ifndef G__OLDIMPLEMENTATION1207
extern G__EXPORT void G__set_smartunload G__P((int smartunload));
#endif

#ifndef G__OLDIMPLEMENTATION1210
typedef int (*G__IgnoreInclude) G__P((const char* fname,const char* expandedfname));
extern G__EXPORT void G__set_ignoreinclude G__P((G__IgnoreInclude ignoreinclude));
#endif

#ifdef G__ASM_WHOLEFUNC
/**************************************************************************
* Interface method to run bytecode function
**************************************************************************/
extern int G__exec_bytecode G__P((G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash));
#endif

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
#define printf  G__printf
#define fprintf G__fprintf
#define fputc   G__fputc
#define putc    G__fputc
#define putchar G__putchar
#define fputs   G__fputs
#define puts    G__puts
#define fgets   G__fgets
#define gets    G__gets

int G__printf G__P((char* fmt,...));
extern G__EXPORT int G__fprintf G__P((FILE* fp,char* fmt,...));
int G__fputc G__P((int character,FILE *fp));
int G__putchar G__P((int character));
int G__fputs G__P((char *string,FILE *fp));
int G__puts G__P((char *string));
char *G__fgets G__P((char *string,int n,FILE *fp));
char *G__gets G__P((char *buffer));

#ifdef G__SPECIALSTDIO

/* THIS IS AN OLD WILDC++ IMPLEMENTATION */
/* signal causes problem in Windows-95 with Tcl/Tk */
#define signal(sigid,p2f)  NULL
#define alarm(time)        NULL

#else /* G__SPECIALSTDIO */

#ifndef G__OLDIMPLEMENTATION614
#ifdef signal
#undef signal
#endif
#define signal G__signal
#define alarm(time)        NULL
typedef void (*G__signaltype)(int,void (*)(int));
G__signaltype G__signal G__P((int sgnl,void (*f)(int)));
extern G__EXPORT int G__setmasksignal G__P((int));
#endif /* ON614 */

#endif /* G__SPECIALSTDIO */

#endif /* WIN32 !SYMANTEC CINTBODY*/
/**************************************************************************
* end of specialstdio or win32
**************************************************************************/

#if defined(__cplusplus) && !defined(__CINT__)
} /* extern C 3 */
#endif

#endif /* __MAKECINT__ */
/**************************************************************************
* endif #ifndef G__MAKECINT
**************************************************************************/

#ifdef G__OLDIMPLEMENTATION1231
/**************************************************************************
* Interface Method type
*
**************************************************************************/
#if defined(__cplusplus) && !defined(__CINT__)
extern "C" { /* extern C 4' */
#endif

#ifdef G__ANSIHEADER
typedef int (*G__InterfaceMethod)(G__value*,G__CONST char*,struct G__param*,int);
#else
typedef int (*G__InterfaceMethod)();
#endif

#if defined(__cplusplus) && !defined(__CINT__)
} /* extern C 4' */
#endif
#endif /* 1231 */


#endif /* G__CI_H */

