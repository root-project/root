/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file g__security.h
 ************************************************************************
 * Description:
 *  cint original extention providing secure C/C++ subset.
 *
 * This file is included from cint core source code too. But dependency is
 * not described in the Makefile. Please touch G__ci.h file whenever changing
 * this file.
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__SECURITY_H
#define G__SECURITY_H

/**************************************************************************
* Security flags
**************************************************************************/
#define G__SECURE_NONE                    0x00000000
#define G__SECURE_LEVEL0                  0x00000007
#define G__SECURE_LEVEL1                  0x000000ff
#define G__SECURE_LEVEL2                  0x00000fff
#define G__SECURE_LEVEL3                  0x0000ffff
#define G__SECURE_LEVEL4                  0x000fffff
#define G__SECURE_LEVEL5                  0x00ffffff
#define G__SECURE_LEVEL6                  0x00ffffff

/* Prohibit indivisual language construct */
#define G__SECURE_EXIT_AT_ERROR            0x7fffffff
/* DON'T USE 0xffffffff. THIS CAUSES PROBLEM WITH DEC ALPHA CXX COMPILER */

/* Cint internal error check */
#define G__SECURE_STACK_DEPTH            0x00000001 /* ifunc.c */
#define G__SECURE_BUFFER_SIZE            0x00000002 /* unused yet */
#define G__SECURE_MARGINAL_CAST          0x00000002 /* added 4/1999 */
#define G__SECURE_POINTER_INIT           0x00000004 /* allocvariable var.c*/
#define G__SECURE_POINTER_TYPE           0x00000008 /* var.c*/

/* Prohibit pointer tricks and goto statement */
#define G__SECURE_POINTER_CALC           0x00000010 /* cast.c opr.c */
#define G__SECURE_CAST2P                 0x00000020 /* var.c cast.c */
#define G__SECURE_GOTO                   0x00000040 /* parse.c */
#define G__SECURE_GARBAGECOLLECTION      0x00000080 /* gcoll.c */

/* Prohibit array referencing by pointer T* a; a[n]; */
#define G__SECURE_POINTER_AS_ARRAY       0x00000100 /* getvariable var.c */

/* Prohibit casting */
#define G__SECURE_CASTING                0x00001000 /* var.c cast.c */

/* More strict limitations */
#define G__SECURE_POINTER_OBJECT         0x00010000 /* unused */
#define G__SECURE_POINTER_INSTANTIATE    0x00020000 /* allocvariable var.c*/
#define G__SECURE_POINTER_REFERENCE      0x00040000 /* getvariable var.c */
#define G__SECURE_POINTER_ASSIGN         0x00080000 /* letvariable var.c */

/* No standard library func */
#define G__SECURE_MALLOC                 0x00100000 /* new.c g__cfunc.c */
#define G__SECURE_FILE_POINTER           0x00200000 /* allocvariable var.c*/
#define G__SECURE_STANDARDLIB            0x00400000 /* g__cfunc.c */

/* not used */
#define G__SECURE_ARRAY                  0x00000000 /* allocvariable var.c*/

/* Preference, invoke command line input at error */
#define G__SECURE_NO_CHANGE              0x10000000 /* pragma.c */
#define G__SECURE_NO_RELAX               0x20000000 /* pragma.c */
#define G__SECURE_PAUSE                  0x40000000 /* error.c */
/* DON'T USE 0x80000000. THIS CAUSES PROBLEM WITH DEC ALPHA CXX COMPILER */

/* Maximum function call stack depth, This limit is just for security. */
#define G__DEFAULT_MAX_STACK_DEPTH    128

#ifdef G__SECURITY

#ifdef G__64BIT
typedef unsigned int G__UINT32 ;
#else
typedef unsigned long G__UINT32 ;
#endif

#ifdef __cplusplus
extern "C" G__UINT32 G__security;
#else
extern G__UINT32 G__security;
#endif

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

#endif
