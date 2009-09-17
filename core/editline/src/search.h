// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: search.fH,v 1.5 2000/09/04 22:06:32 lukem Exp $	*/

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
 * el.search.fH: Line and history searching utilities
 */
#ifndef _h_el_search
#define _h_el_search

#include "histedit.h"

struct ElSearch_t {
   char* fPatBuf;                        /* The pattern buffer		*/
   size_t fPatLen;                       /* Length of the pattern buffer	*/
   int fPatDir;                          /* Direction of the last search	*/
   int fChaDir;                          /* Character search direction	*/
   char fChaCha;                         /* Character we are looking for	*/
};


el_protected int el_match(const char*, const char*);
el_protected int search_init(EditLine_t*);
el_protected void search_end(EditLine_t*);
el_protected int c_hmatch(EditLine_t*, const char*);
el_protected void c_setpat(EditLine_t*);
el_protected ElAction_t ce_inc_search(EditLine_t*, int);
el_protected ElAction_t cv_search(EditLine_t*, int);
el_protected ElAction_t ce_search_line(EditLine_t*, char*, int);
el_protected ElAction_t cv_repeat_srch(EditLine_t*, int);
el_protected ElAction_t cv_csearch_back(EditLine_t*, int, int, int);
el_protected ElAction_t cv_csearch_fwd(EditLine_t*, int, int, int);

#endif /* _h_el_search */
