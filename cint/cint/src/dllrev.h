/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file dllrev.h
 ************************************************************************
 * Description:
 *  Dynamic Link Library revision
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__DLLREV_H
#define G__DLLREV_H

/**************************************************************************
* Dynamic Link Library revision
**************************************************************************/


#ifdef G__VARIABLEFPOS
#define G__VF 10000000
#else
#define G__VF 0
#endif

#ifdef G__TYPEDEFFPOS
#define G__TF 20000000
#else
#define G__TF 0
#endif

#ifdef G__ROOT
/* #define G__RT 40000000 */
#define G__RT 0
#else
#define G__RT 0
#endif

#define G__CREATEDLLREV       (51515+G__VF+G__TF+G__RT)
#define G__ACCEPTDLLREV_FROM  (51501+G__VF+G__TF+G__RT)
#define G__ACCEPTDLLREV_UPTO  (51599+G__VF+G__TF+G__RT)

#endif
