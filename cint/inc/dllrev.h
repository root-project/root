/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file dllrev.h
 ************************************************************************
 * Description:
 *  Dynamic Link Library revision
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#ifndef G__DLLREV_H
#define G__DLLREV_H

/**************************************************************************
* Dynamic Link Library revision
**************************************************************************/

#ifndef G__OLDIMPLEMENTATION1169

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

#ifndef G__OLDIMPLEMENTATION1530
#define G__CREATEDLLREV       (51503+G__VF+G__TF)
#define G__ACCEPTDLLREV_FROM  (51501+G__VF+G__TF)
#define G__ACCEPTDLLREV_UPTO  (51599+G__VF+G__TF)
#else
  #ifdef G__CONSTNESSFLAG
  #define G__CREATEDLLREV       (51472+G__VF+G__TF)
  #define G__ACCEPTDLLREV_FROM  (51111+G__VF+G__TF)
  #define G__ACCEPTDLLREV_UPTO  (51472+G__VF+G__TF)
  #else
  #define G__CREATEDLLREV       (51111+G__VF+G__TF)
  #define G__ACCEPTDLLREV_FROM  (51111+G__VF+G__TF)
  #define G__ACCEPTDLLREV_UPTO  (51415+G__VF+G__TF)
  #endif
#endif

#else

#define G__DLLREV  51111

#endif

#endif
