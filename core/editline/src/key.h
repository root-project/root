// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: key.fH,v 1.5 2001/01/23 15:55:30 jdolecek Exp $	*/

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
 * el.fKey.fH: Key macro header
 */
#ifndef _h_el_key
#define _h_el_key

typedef union KeyValue_t {
   ElAction_t fCmd;             /* If it is a command the #	*/
   char* fStr;                   /* If it is a string...		*/
} KeyValue_t;

typedef struct KeyNode_t KeyNode_t;

typedef struct ElKey_t {
   char* fBuf;                   /* Key print buffer		*/
   KeyNode_t* fMap;             /* Key map			*/
   KeyValue_t fVal;             /* Local conversion buffer	*/
} ElKey_t;

#define XK_CMD 0
#define XK_STR 1
#define XK_NOD 2
#define XK_EXE 3

el_protected int key_init(EditLine_t*);
el_protected void key_end(EditLine_t*);
el_protected KeyValue_t* key_map_cmd(EditLine_t*, int);
el_protected KeyValue_t* key_map_str(EditLine_t*, char*);
el_protected void key_reset(EditLine_t*);
el_protected int key_get(EditLine_t*, char*, KeyValue_t*);
el_protected void key_add(EditLine_t*, const char*, KeyValue_t*, int);
el_protected void key_clear(EditLine_t*, ElAction_t*, char*);
el_protected int key_delete(EditLine_t*, char*);
el_protected void key_print(EditLine_t*, const char*);
el_protected void key_kprint(EditLine_t*, const char*, KeyValue_t*, int);
el_protected char* key__decode_str(char*, char*, const char*);

#endif /* _h_el_key */
