// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: hist.c,v 1.9 2001/05/17 01:02:17 christos Exp $	*/

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

#include "compat.h"

/*
 * hist.c: History_t access functions
 */
#include "sys.h"
#include <stdlib.h>
#include "el.h"

/* hist_init():
 *	Initialization function.
 */
el_protected int
hist_init(EditLine_t* el) {
   el->fHistory.fFun = NULL;
   el->fHistory.fRef = NULL;
   el->fHistory.fBuf = (char*) el_malloc(EL_BUFSIZ);
   el->fHistory.fSz = EL_BUFSIZ;

   if (el->fHistory.fBuf == NULL) {
      return -1;
   }
   el->fHistory.fLast = el->fHistory.fBuf;
   return 0;
}


/* hist_end():
 *	clean up history;
 */
el_protected void
hist_end(EditLine_t* el) {
   el_free((ptr_t) el->fHistory.fBuf);
   el->fHistory.fBuf = NULL;
}


/* hist_set():
 *	Set new history interface
 */
el_protected int
hist_set(EditLine_t* el, HistFun_t fun, ptr_t ptr) {
   el->fHistory.fRef = ptr;
   el->fHistory.fFun = fun;
   return 0;
}


/* hist_get():
 *	Get a history line and update it in the buffer.
 *	eventno tells us the event to get.
 */
el_protected ElAction_t
hist_get(EditLine_t* el) {
   const char* hp;
   int h;

   if (el->fHistory.fEventNo == 0) {           /* if really the current line */
      (void) strncpy(el->fLine.fBuffer, el->fHistory.fBuf,
                     el->fHistory.fSz);
      ElColor_t* col = el->fLine.fBufColor;

      for (size_t i = 0; i < (size_t) el->fHistory.fSz; ++i) {
         col[i] = -1;
      }
      el->fLine.fLastChar = el->fLine.fBuffer +
                             (el->fHistory.fLast - el->fHistory.fBuf);

#ifdef KSHVI

      if (el->fMap.fType == MAP_VI) {
         el->fLine.fCursor = el->fLine.fBuffer;
      } else
#endif /* KSHVI */
      el->fLine.fCursor = el->fLine.fLastChar;

      return CC_REFRESH;
   }

   if (el->fHistory.fRef == NULL) {
      return CC_ERROR;
   }

   hp = HIST_FIRST(el);

   if (hp == NULL) {
      return CC_ERROR;
   }

   for (h = 1; h < el->fHistory.fEventNo; h++) {
      if ((hp = HIST_NEXT(el)) == NULL) {
         el->fHistory.fEventNo = h;
         return CC_ERROR;
      }
   }
   (void) strncpy(el->fLine.fBuffer, hp,
                  (size_t) (el->fLine.fLimit - el->fLine.fBuffer));
   ElColor_t* col = el->fLine.fBufColor;

   for (size_t i = 0; i < (size_t) (el->fLine.fLimit - el->fLine.fBuffer); ++i) {
      col[i] = -1;
   }
   el->fLine.fLastChar = el->fLine.fBuffer + strlen(el->fLine.fBuffer);

   if (el->fLine.fLastChar > el->fLine.fBuffer) {
      if (el->fLine.fLastChar[-1] == '\n') {
         el->fLine.fLastChar--;
      }

      if (el->fLine.fLastChar[-1] == ' ') {
         el->fLine.fLastChar--;
      }

      if (el->fLine.fLastChar < el->fLine.fBuffer) {
         el->fLine.fLastChar = el->fLine.fBuffer;
      }
   }
#ifdef KSHVI

   if (el->fMap.fType == MAP_VI) {
      el->fLine.fCursor = el->fLine.fBuffer;
   } else
#endif /* KSHVI */
   el->fLine.fCursor = el->fLine.fLastChar;

   return CC_REFRESH;
} // hist_get


/* hist_list()
 *	List history entries
 */
el_protected int
/*ARGSUSED*/
hist_list(EditLine_t* el, int /*argc*/, const char** /*argv*/) {
   const char* str;

   if (el->fHistory.fRef == NULL) {
      return -1;
   }

   for (str = HIST_LAST(el); str != NULL; str = HIST_PREV(el)) {
      (void) fprintf(el->fOutFile, "%d %s%s",
                     el->fHistory.fEv.fNum, str,
                     (NULL == strstr(str, "\n")) ? "\n" : ""

                     /* ^^^
                        added by stephan@s11n.net to fix
                        when used with GNU RL compat mode,
                        which strips the newline (which is
                        more sane, IMO).
                      */
      );
   }

   return 0;
} // hist_list


/* hist_enlargebuf()
 *	Enlarge history buffer to specified value. Called from el_enlargebufs().
 *	Return 0 for failure, 1 for success.
 */
el_protected int
/*ARGSUSED*/
hist_enlargebuf(EditLine_t* el, size_t oldsz, size_t newsz) {
   char* newbuf;

   newbuf = (char*) realloc(el->fHistory.fBuf, newsz);

   if (!newbuf) {
      return 0;
   }

   (void) memset(&newbuf[oldsz], '\0', newsz - oldsz);

   el->fHistory.fLast = newbuf +
                         (el->fHistory.fLast - el->fHistory.fBuf);
   el->fHistory.fBuf = newbuf;
   el->fHistory.fSz = newsz;

   return 1;
} // hist_enlargebuf
