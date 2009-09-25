// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: term.fH,v 1.12 2001/01/04 15:56:32 christos Exp $	*/

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
 * el.term.fH: Termcap header
 */
#ifndef _h_el_term
#define _h_el_term

#include "histedit.h"

typedef struct {                /* Symbolic function key bindings	*/
   const char* fName;                  /* name of the key			*/
   int fKey;                     /* Index in termcap table		*/
   KeyValue_t fFun;             /* Function bound to it			*/
   int fType;                    /* Type of function			*/
} FKey_t;

typedef struct {
   ElCoord_t fSize;                      /* # lines and cols	*/
   int fFlags;
#define TERM_CAN_INSERT 0x001           /* Has insert cap	*/
#define TERM_CAN_DELETE 0x002           /* Has delete cap	*/
#define TERM_CAN_CEOL 0x004             /* Has CEOL cap		*/
#define TERM_CAN_TAB 0x008              /* Can use tabs		*/
#define TERM_CAN_ME 0x010               /* Can turn all attrs.	*/
#define TERM_CAN_UP 0x020               /* Can move up		*/
#define TERM_HAS_META 0x040             /* Has a meta key	*/
#define TERM_HAS_AUTO_MARGINS 0x080     /* Has auto margins	*/
#define TERM_HAS_MAGIC_MARGINS 0x100    /* Has magic margins	*/
   char* fBuf;                         /* Termcap buffer	*/
   int fLoc;                           /* location used	*/
   char** fStr;                        /* termcap strings	*/
   int* fVal;                          /* termcap values	*/
   char* fCap;                         /* Termcap buffer	*/
   FKey_t* fFKey;                      /* Array of keys	*/
} ElTerm_t;

/*
 * fKey indexes
 */
#define A_K_DN 0
#define A_K_UP 1
#define A_K_LT 2
#define A_K_RT 3
#define A_K_HO 4
#define A_K_EN 5
#define A_K_DE 6
#define A_K_NKEYS 7

el_protected void term_move_to_line(EditLine_t*, int);
el_protected void term_move_to_char(EditLine_t*, int);
el_protected void term_clear_EOL(EditLine_t*, int);
el_protected void term_overwrite(EditLine_t*, const char*, ElColor_t*, int);
el_protected void term_insertwrite(EditLine_t*, const char*, ElColor_t*, int);
el_protected void term_deletechars(EditLine_t*, int);
el_protected void term_clear_screen(EditLine_t*);
el_protected void term_beep(EditLine_t*);
el_protected int term_change_size(EditLine_t*, int, int);
el_protected int term_get_size(EditLine_t*, int*, int*);
el_protected int term_init(EditLine_t*);
el_protected void term_bind_arrow(EditLine_t*);
el_protected void term_print_arrow(EditLine_t*, const char*);
el_protected int term_clear_arrow(EditLine_t*, char*);
el_protected int term_set_arrow(EditLine_t*, char*, KeyValue_t*, int);
el_protected void term_end(EditLine_t*);
el_protected int term_set(EditLine_t*, const char*);
el_protected int term_settc(EditLine_t*, int, const char**);
el_protected int term_telltc(EditLine_t*, int, const char**);
el_protected int term_echotc(EditLine_t*, int, const char**);
el_protected extern "C" int term__putc(int);
el_protected int term__putcolorch(int, ElColor_t*);
el_protected void term__setcolor(int fgcol);
el_protected int  term__atocolor(const char* name);
el_protected void term__resetcolor(void);
el_protected void term__repaint(EditLine_t* el, int index);
el_protected void term__flush(void);

/*
 * Easy access macros
 */
#define EL_FLAGS (el)->fTerm.fFlags

#define EL_CAN_UP (EL_FLAGS & TERM_CAN_UP)
#define EL_CAN_INSERT (EL_FLAGS & TERM_CAN_INSERT)
#define EL_CAN_DELETE (EL_FLAGS & TERM_CAN_DELETE)
#define EL_CAN_CEOL (EL_FLAGS & TERM_CAN_CEOL)
#define EL_CAN_TAB (EL_FLAGS & TERM_CAN_TAB)
#define EL_CAN_ME (EL_FLAGS & TERM_CAN_ME)
#define EL_HAS_META (EL_FLAGS & TERM_HAS_META)
#define EL_HAS_AUTO_MARGINS (EL_FLAGS & TERM_HAS_AUTO_MARGINS)
#define EL_HAS_MAGIC_MARGINS (EL_FLAGS & TERM_HAS_MAGIC_MARGINS)

#endif /* _h_el_term */
