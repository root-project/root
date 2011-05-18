// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: chared.fH,v 1.6 2001/01/10 07:45:41 jdolecek Exp $	*/

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
 * el.chared.fH: Character editor interface
 */
#ifndef _h_el_chared
#define _h_el_chared

#include <ctype.h>
#include <string.h>

#include "histedit.h"

#define EL_MAXMACRO 10

/*
 * This is a issue of basic "vi" look-and-feel. Defining VI_MOVE works
 * like real vi: i.e. the transition from command<->insert modes moves
 * the cursor.
 *
 * On the other hand we really don't want to move the cursor, because
 * all the editing commands don't include the character under the cursor.
 * Probably the best fix is to make all the editing commands aware of
 * this fact.
 */
#define VI_MOVE


typedef struct CMacro_t {
   int fLevel;
   char** fMacro;
   char* fNLine;
} CMacro_t;

/*
 * Undo information for both vi and emacs
 */
typedef struct CUndo_t {
   int fAction;
   size_t fISize;
   size_t fDSize;
   char* fPtr;
   char* fBuf;
} CUndo_t;

/*
 * Current action information for vi
 */
typedef struct CVCmd_t {
   int fAction;
   char* fPos;
   char* fIns;
} CVCmd_t;

/*
 * Kill buffer for emacs
 */
typedef struct CKill_t {
   char* fBuf;
   char* fLast;
   char* fMark;
} CKill_t;

/*
 * Note that we use both data structures because the user can bind
 * commands from both editors!
 */
typedef struct ElCharEd_t {
   CUndo_t fUndo;
   CKill_t fKill;
   CVCmd_t fVCmd;
   CMacro_t fMacro;
} ElCharEd_t;


#define STReof "^D\b\b"
#define STRQQ "\"\""

#define isglob(a) (strchr("*[]?", (a)) != NULL)
#define isword(a) (isprint(a))

#define NOP 0x00
#define DELETE 0x01
#define INSERT 0x02
#define CHANGE 0x04

#define CHAR_FWD 0
#define CHAR_BACK 1

#define MODE_INSERT 0
#define MODE_REPLACE 1
#define MODE_REPLACE_1 2

#include "common.h"
#ifdef EL_USE_VI
# include "vi.h"
#endif
#include "emacs.h"
#include "search.h"
#include "fcns.h"


el_protected int cv__isword(int);
el_protected void cv_delfini(EditLine_t*);
el_protected char* cv__endword(char*, char*, int);
el_protected int ce__isword(int);
el_protected void cv_undo(EditLine_t *, int, size_t, char*);
el_protected char* cv_next_word(EditLine_t *, char*, char*, int, int(*) (int));
el_protected char* cv_prev_word(EditLine_t *, char*, char*, int, int(*) (int));
el_protected char* c__next_word(char*, char*, int, int(*) (int));
el_protected char* c__prev_word(char*, char*, int, int(*) (int));
el_protected void c_insert(EditLine_t*, int);
el_protected void c_delbefore(EditLine_t*, int);
el_protected void c_delafter(EditLine_t*, int);
el_protected int c_gets(EditLine_t*, char*);
el_protected int c_hpos(EditLine_t*);

el_protected int ch_init(EditLine_t*);
el_protected void ch_reset(EditLine_t*);
/* el_protected int	 ch_enlargebufs	__P((EditLine_t *, size_t)); */
el_protected int ch_enlargebufs(EditLine_t *, size_t);
el_protected void ch_end(EditLine_t*);

#endif /* _h_el_chared */
