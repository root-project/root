// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: el.fH,v 1.8 2001/01/06 14:44:50 jdolecek Exp $	*/

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
 * el.fH: Internal structures.
 */
#ifndef _h_el
#define _h_el

/*
 * Local defaults
 */
#define KSHVI
/* #define	VIDEFAULT */

#include <stdio.h>
#include <sys/types.h>

#define EL_BUFSIZ 1024                  /* Maximum line size		*/

#define HANDLE_SIGNALS 1 << 0
#define NO_TTY 1 << 1
#define EDIT_DISABLED 1 << 2
// #define DEBUG_READ 1

typedef int ElBool_t;                     /* True or not			*/

typedef unsigned char ElAction_t;      /* Index to command array	*/

struct ElCoord_t {                /* Position on the screen	*/
   int fH;
   int fV;
};

struct ElColor_t {
   int fForeColor;                       /* The foreground text colour */
   int fBackColor;                       /* The background colour */
   ElColor_t(int f = -1, int b = -2): fForeColor(f),
      fBackColor(b) {}

   ElColor_t&
   operator =(const ElColor_t& color) {
      (*this).fForeColor = color.fForeColor;
      (*this).fBackColor = color.fBackColor;
      return *this;
   }


   ElColor_t&
   operator =(int val) {
      fForeColor = val;
      fBackColor = val;
      return *this;
   }
};


struct ElLine_t {
   char* fBuffer;                                /* Input line			*/
   ElColor_t* fBufColor;                         /* Color info for each char in buffer */
   char* fCursor;                                /* Cursor position		*/
   char* fLastChar;                              /* Last character		*/
   const char* fLimit;                           /* Max position		*/
};

/*
 * Editor state
 */
struct ElState_t {
   int fInputMode;                       /* What mode are we in?		*/
   int fDoingArg;                        /* Are we getting an argument?	*/
   int fArgument;                        /* Numeric argument		*/
   int fMetaNext;                        /* Is the next char a meta char */
   ElAction_t fLastCmd;                 /* Previous command		*/
   int fReplayHist;                      /* Previous command		*/
};

/*
 * Until we come up with something better...
 */
#define el_malloc(a) malloc(a)
#define el_realloc(a, b) realloc(a, b)
#define el_free(a) free(a)

#include "compat.h"
#include "sys.h"
#include "tty.h"
#include "prompt.h"
#include "key.h"
#include "term.h"
#include "refresh.h"
#include "chared.h"
#include "common.h"
#include "search.h"
#include "hist.h"
#include "map.h"
#include "parse.h"
#include "sig.h"
#include "help.h"

struct EditLine_t {
   char* fProg;                       /* the program name		*/
   FILE* fOutFile;                    /* Stdio stuff			*/
   FILE* fErrFile;                    /* Stdio stuff			*/
   FILE* fIn;                         /* Input file if !tty     */
   int fInFD;                         /* Input file descriptor	*/
   int fFlags;                        /* Various flags.		*/
   ElCoord_t fCursor;                 /* Cursor location		*/
   char** fDisplay;                   /* Real screen image = what is there */
   ElColor_t** fDispColor;            /* Color for each char in fDisplay */
   char** fVDisplay;                  /* Virtual screen image = what we see */
   ElColor_t** fVDispColor;           /* Color for each char in fVDisplay*/
   ElLine_t fLine;                   /* The current line information	*/
   ElState_t fState;                 /* Current editor state		*/
   ElTerm_t fTerm;                   /* Terminal dependent stuff	*/
   ElTTY_t fTTY;                     /* Tty dependent stuff		*/
   ElRefresh_t fRefresh;             /* Refresh stuff		*/
   ElPrompt_t fPrompt;               /* Prompt stuff			*/
   ElPrompt_t fRPrompt;              /* Prompt stuff			*/
   ElCharEd_t fCharEd;               /* Characted editor stuff	*/
   ElMap_t fMap;                     /* Key mapping stuff		*/
   ElKey_t fKey;                     /* Key binding stuff		*/
   ElHistory_t fHistory;             /* History_t stuff		*/
   ElSearch_t fSearch;               /* Search stuff			*/
   ElSignal_t fSignal;               /* Signal handling stuff	*/
};

el_protected int el_editmode(EditLine_t*, int, const char**);

/**
   Added by stephan@s11n.net: returns the editline object associated
   with the readline compatibility interface, to allow clients to
   customize that object further.
 */
el_public EditLine_t* el_readline_el();


#ifdef DEBUG
# define EL_ABORT(a) (void) (fprintf(el->fErrFile, "%s, %d: ", \
                                     __FILE__, __LINE__), fprintf a, abort())
#else
# define EL_ABORT(a) abort()
#endif
#endif /* _h_el */
