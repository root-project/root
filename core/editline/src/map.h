// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: map.fH,v 1.6 2001/01/09 17:22:09 jdolecek Exp $	*/

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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
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
 * el.fMap.fH:	Editor maps
 */
#ifndef _h_el_map
#define _h_el_map

typedef struct ElBindings_t {  /* for the "bind" shell command */
   const char* fName;                    /* function name for bind command */
   int fFunc;                            /* function numeric value */
   const char* fDescription;             /* description of function */
} ElBindings_t;


typedef struct ElMap_t {
   ElAction_t* fAlt;                    /* The current alternate key map */
   ElAction_t* fKey;                    /* The current normal key map	*/
   ElAction_t* fCurrent;                /* The keymap we are using	*/
   const ElAction_t* fEmacs;            /* The default emacs key map	*/
   const ElAction_t* fVic;              /* The vi command mode key map	*/
   const ElAction_t* fVii;              /* The vi insert mode key map	*/
   int fType;                            /* Emacs or vi			*/
   ElBindings_t* fHelp;                 /* The help for the editor functions */
   ElFunc_t* fFunc;                     /* List of available functions	*/
   int fNFunc;                           /* The number of functions/help items */
} ElMap_t;

#define MAP_EMACS 0
#define MAP_VI 1

el_protected int map_bind(EditLine_t*, int, const char**);
el_protected int map_init(EditLine_t*);
el_protected void map_end(EditLine_t*);
el_protected void map_init_vi(EditLine_t*);
el_protected void map_init_emacs(EditLine_t*);
el_protected int map_set_editor(EditLine_t*, char*);
el_protected int map_get_editor(EditLine_t*, const char**);
el_protected int map_addfunc(EditLine_t *, const char*, const char*, ElFunc_t);

#endif /* _h_el_map */
