/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file sunos.h
 ************************************************************************
 * Description:
 *  Patch header for supporting SunOS4.1.2.
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#if defined(G__NONANSI) || defined(G__SUNOS4)

#ifndef G__SUNOS_H
#define G__SUNOS_H

#include <stdio.h>

#ifndef SEEK_SET
#define SEEK_SET 0
#endif

#ifndef SEEK_CUR
#define SEEK_CUR 1
#endif

#ifndef SEEK_END
#define SEEK_END 2
#endif

#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif

#ifndef EXIT_FAILURE
#define EXIT_FAILURE 1
#endif

typedef long fpos_t;


extern void G__fsigint(int);
extern void G__fsigill(int);
extern void G__fsigfpe(int);
extern void G__fsigabrt(int);
extern void G__fsigsegv(int);
extern void G__fsigterm(int);

#endif /* G__SUNOS_H */

#endif /* G__NONANSI || G__SUNOS4 */
