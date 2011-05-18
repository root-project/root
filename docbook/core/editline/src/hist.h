// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: hist.fH,v 1.6 2001/01/10 07:45:41 jdolecek Exp $	*/

/*-
 * Copyright (c) 1992, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * This code is derived from software contributed to Berkeley by
 * Christos Zoulas of Cornell University.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

/*
 * el.hist.c: History_t functions
 */
#ifndef _h_el_hist
#define _h_el_hist

#include "histedit.h"

typedef int (*HistFun_t)(ptr_t, HistEvent_t*, int, ...);

typedef struct ElHistory_t {
   char* fBuf;                           /* The history buffer		*/
   size_t fSz;                           /* Size of history buffer	*/
   char* fLast;                          /* The last character		*/
   int fEventNo;                         /* Event we are looking for	*/
   ptr_t fRef;                           /* Argument for history fcns	*/
   HistFun_t fFun;                      /* Event access			*/
   HistEvent_t fEv;                        /* Event cookie			*/
} ElHistory_t;

#define HIST_FUN(el, fn, arg) \
   ((((*(el)->fHistory.fFun)((el)->fHistory.fRef, &(el)->fHistory.fEv, \
                              fn, arg)) == -1) ? NULL : (el)->fHistory.fEv.fStr)

#define HIST_NEXT(el) HIST_FUN(el, H_NEXT, NULL)
#define HIST_FIRST(el) HIST_FUN(el, H_FIRST, NULL)
#define HIST_LAST(el) HIST_FUN(el, H_LAST, NULL)
#define HIST_PREV(el) HIST_FUN(el, H_PREV, NULL)
#define HIST_EVENT(el, num) HIST_FUN(el, H_EVENT, num)
#define HIST_LOAD(el, fname) HIST_FUN(el, H_LOAD fname)
#define HIST_SAVE(el, fname) HIST_FUN(el, H_SAVE fname)

el_protected int hist_init(EditLine_t*);
el_protected void hist_end(EditLine_t*);
el_protected ElAction_t hist_get(EditLine_t*);
el_protected int hist_set(EditLine_t *, HistFun_t, ptr_t);
el_protected int hist_list(EditLine_t*, int, const char**);
el_protected int hist_enlargebuf(EditLine_t *, size_t, size_t);

#endif /* _h_el_hist */
