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

#ifdef G__CONSTNESSFLAG
#define G__CREATEDLLREV  51428
#define G__ACCEPTDLLREV_FROM  51111  /* should be 51428 */
#define G__ACCEPTDLLREV_UPTO  51428
#else
#define G__CREATEDLLREV  51111
#define G__ACCEPTDLLREV_FROM  51111
#define G__ACCEPTDLLREV_UPTO  51415
#endif

#else

#define G__DLLREV  51111

#endif

#endif
