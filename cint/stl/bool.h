/*
 *
 * Copyright (c) 1994
 * Hewlett-Packard Company
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Hewlett-Packard Company makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 *
 */

#ifdef __CINT__
enum bool { 
#ifndef FALSE
  FALSE = 0,
#endif
#ifndef TRUE
  TRUE = 1, 
#endif
  false = 0, true = 1 };
#else
#define bool int
#define true 1
#define false 0
#endif
