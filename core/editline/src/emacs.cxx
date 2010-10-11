// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: emacs.c,v 1.9 2001/01/10 07:45:41 jdolecek Exp $	*/

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
 * emacs.c: Emacs functions
 */
#include "sys.h"
#include "el.h"

/* em_delete_or_list():
 *	Delete character under cursor or list completions if at end of line
 *	[^D]
 */
el_protected ElAction_t
/*ARGSUSED*/
em_delete_or_list(EditLine_t* el, int /*c*/) {
   if (el->fLine.fCursor == el->fLine.fLastChar) {
      /* if I'm at the end */
      if (el->fLine.fCursor == el->fLine.fBuffer) {
         /* and the beginning */
         term_overwrite(el, STReof, 0, 4);                      /* then do a EOF */
         term__flush();
         return CC_EOF;
      } else {
         /*
          * Here we could list completions, but it is an
          * error right now
          */
         term_beep(el);
         return CC_ERROR;
      }
   } else {
      c_delafter(el, el->fState.fArgument);            /* delete after dot */

      if (el->fLine.fCursor > el->fLine.fLastChar) {
         el->fLine.fCursor = el->fLine.fLastChar;
      }
      /* bounds check */
      return CC_REFRESH;
   }
} // em_delete_or_list


/* em_delete_next_word():
 *	Cut from cursor to end of current word
 *	[M-d]
 */
el_protected ElAction_t
/*ARGSUSED*/
em_delete_next_word(EditLine_t* el, int /*c*/) {
   char* cp, * p, * kp;

   if (el->fLine.fCursor == el->fLine.fLastChar) {
      return CC_ERROR;
   }

   cp = c__next_word(el->fLine.fCursor, el->fLine.fLastChar,
                     el->fState.fArgument, ce__isword);

   for (p = el->fLine.fCursor, kp = el->fCharEd.fKill.fBuf; p < cp; p++) {
      /* save the text */
      *kp++ = *p;
   }
   el->fCharEd.fKill.fLast = kp;

   c_delafter(el, cp - el->fLine.fCursor);             /* delete after dot */

   if (el->fLine.fCursor > el->fLine.fLastChar) {
      el->fLine.fCursor = el->fLine.fLastChar;
   }
   /* bounds check */
   return CC_REFRESH;
} // em_delete_next_word


/* em_yank():
 *	Paste cut buffer at cursor position
 *	[^Y]
 */
el_protected ElAction_t
/*ARGSUSED*/
em_yank(EditLine_t* el, int /*c*/) {
   char* kp, * cp;

   if (el->fCharEd.fKill.fLast == el->fCharEd.fKill.fBuf) {
      if (!ch_enlargebufs(el, 1)) {
         return CC_ERROR;
      }
   }

   if (el->fLine.fLastChar +
       (el->fCharEd.fKill.fLast - el->fCharEd.fKill.fBuf) >=
       el->fLine.fLimit) {
      return CC_ERROR;
   }

   el->fCharEd.fKill.fMark = el->fLine.fCursor;
   cp = el->fLine.fCursor;

   /* open the space, */
   c_insert(el, el->fCharEd.fKill.fLast - el->fCharEd.fKill.fBuf);

   /* copy the chars */
   for (kp = el->fCharEd.fKill.fBuf; kp < el->fCharEd.fKill.fLast; kp++) {
      *cp++ = *kp;
   }

   /* if an arg, cursor at beginning else cursor at end */
   if (el->fState.fArgument == 1) {
      el->fLine.fCursor = cp;
   }

   return CC_REFRESH;
} // em_yank


/* em_kill_line():
 *	Cut the entire line and save in cut buffer
 *	[^U]
 */
el_protected ElAction_t
/*ARGSUSED*/
em_kill_line(EditLine_t* el, int /*c*/) {
   char* kp, * cp;

   cp = el->fLine.fBuffer;
   kp = el->fCharEd.fKill.fBuf;

   while (cp < el->fLine.fLastChar)
      *kp++ = *cp++;            /* copy it */
   el->fCharEd.fKill.fLast = kp;
   /* zap! -- delete all of it */
   el->fLine.fLastChar = el->fLine.fBuffer;
   el->fLine.fCursor = el->fLine.fBuffer;
   return CC_REFRESH;
}


/* em_kill_region():
 *	Cut area between mark and cursor and save in cut buffer
 *	[^W]
 */
el_protected ElAction_t
/*ARGSUSED*/
em_kill_region(EditLine_t* el, int /*c*/) {
   char* kp, * cp;

   if (!el->fCharEd.fKill.fMark) {
      return CC_ERROR;
   }

   if (el->fCharEd.fKill.fMark > el->fLine.fCursor) {
      cp = el->fLine.fCursor;
      kp = el->fCharEd.fKill.fBuf;

      while (cp < el->fCharEd.fKill.fMark)
         *kp++ = *cp++;                 /* copy it */
      el->fCharEd.fKill.fLast = kp;
      c_delafter(el, cp - el->fLine.fCursor);
   } else {                     /* mark is before cursor */
      cp = el->fCharEd.fKill.fMark;
      kp = el->fCharEd.fKill.fBuf;

      while (cp < el->fLine.fCursor)
         *kp++ = *cp++;                 /* copy it */
      el->fCharEd.fKill.fLast = kp;
      c_delbefore(el, cp - el->fCharEd.fKill.fMark);
      el->fLine.fCursor = el->fCharEd.fKill.fMark;
   }
   return CC_REFRESH;
} // em_kill_region


/* em_copy_region():
 *	Copy area between mark and cursor to cut buffer
 *	[M-W]
 */
el_protected ElAction_t
/*ARGSUSED*/
em_copy_region(EditLine_t* el, int /*c*/) {
   char* kp, * cp;

   if (!el->fCharEd.fKill.fMark) {
      return CC_ERROR;
   }

   if (el->fCharEd.fKill.fMark > el->fLine.fCursor) {
      cp = el->fLine.fCursor;
      kp = el->fCharEd.fKill.fBuf;

      while (cp < el->fCharEd.fKill.fMark)
         *kp++ = *cp++;                 /* copy it */
      el->fCharEd.fKill.fLast = kp;
   } else {
      cp = el->fCharEd.fKill.fMark;
      kp = el->fCharEd.fKill.fBuf;

      while (cp < el->fLine.fCursor)
         *kp++ = *cp++;                 /* copy it */
      el->fCharEd.fKill.fLast = kp;
   }
   return CC_NORM;
} // em_copy_region


/* em_gosmacs_traspose():
 *	Exchange the two characters before the cursor
 *	Gosling emacs transpose chars [^T]
 */
el_protected ElAction_t
em_gosmacs_traspose(EditLine_t* el, int c) {
   if (el->fLine.fCursor > &el->fLine.fBuffer[1]) {
      /* must have at least two chars entered */
      c = el->fLine.fCursor[-2];
      el->fLine.fCursor[-2] = el->fLine.fCursor[-1];
      el->fLine.fCursor[-1] = c;
      return CC_REFRESH;
   } else {
      return CC_ERROR;
   }
}


/* em_next_word():
 *	Move next to end of current word
 *	[M-f]
 */
el_protected ElAction_t
/*ARGSUSED*/
em_next_word(EditLine_t* el, int /*c*/) {
   if (el->fLine.fCursor == el->fLine.fLastChar) {
      return CC_ERROR;
   }

   el->fLine.fCursor = c__next_word(el->fLine.fCursor,
                                     el->fLine.fLastChar,
                                     el->fState.fArgument,
                                     ce__isword);

   if (el->fMap.fType == MAP_VI) {
      if (el->fCharEd.fVCmd.fAction & DELETE) {
         cv_delfini(el);
         return CC_REFRESH;
      }
   }
   return CC_CURSOR;
} // em_next_word


/* em_upper_case():
 *	Uppercase the characters from cursor to end of current word
 *	[M-u]
 */
el_protected ElAction_t
/*ARGSUSED*/
em_upper_case(EditLine_t* el, int /*c*/) {
   char* cp, * ep;

   ep = c__next_word(el->fLine.fCursor, el->fLine.fLastChar,
                     el->fState.fArgument, ce__isword);

   for (cp = el->fLine.fCursor; cp < ep; cp++) {
      if (islower((unsigned char) *cp)) {
         *cp = toupper(*cp);
      }
   }

   el->fLine.fCursor = ep;

   if (el->fLine.fCursor > el->fLine.fLastChar) {
      el->fLine.fCursor = el->fLine.fLastChar;
   }
   return CC_REFRESH;
} // em_upper_case


/* em_capitol_case():
 *	Capitalize the characters from cursor to end of current word
 *	[M-c]
 */
el_protected ElAction_t
/*ARGSUSED*/
em_capitol_case(EditLine_t* el, int /*c*/) {
   char* cp, * ep;

   ep = c__next_word(el->fLine.fCursor, el->fLine.fLastChar,
                     el->fState.fArgument, ce__isword);

   for (cp = el->fLine.fCursor; cp < ep; cp++) {
      if (isalpha((unsigned char) *cp)) {
         if (islower((unsigned char) *cp)) {
            *cp = toupper(*cp);
         }
         cp++;
         break;
      }
   }

   for ( ; cp < ep; cp++) {
      if (isupper((unsigned char) *cp)) {
         *cp = tolower(*cp);
      }
   }

   el->fLine.fCursor = ep;

   if (el->fLine.fCursor > el->fLine.fLastChar) {
      el->fLine.fCursor = el->fLine.fLastChar;
   }
   return CC_REFRESH;
} // em_capitol_case


/* em_lower_case():
 *	Lowercase the characters from cursor to end of current word
 *	[M-l]
 */
el_protected ElAction_t
/*ARGSUSED*/
em_lower_case(EditLine_t* el, int /*c*/) {
   char* cp, * ep;

   ep = c__next_word(el->fLine.fCursor, el->fLine.fLastChar,
                     el->fState.fArgument, ce__isword);

   for (cp = el->fLine.fCursor; cp < ep; cp++) {
      if (isupper((unsigned char) *cp)) {
         *cp = tolower(*cp);
      }
   }

   el->fLine.fCursor = ep;

   if (el->fLine.fCursor > el->fLine.fLastChar) {
      el->fLine.fCursor = el->fLine.fLastChar;
   }
   return CC_REFRESH;
} // em_lower_case


/* em_set_mark():
 *	Set the mark at cursor
 *	[^@]
 */
el_protected ElAction_t
/*ARGSUSED*/
em_set_mark(EditLine_t* el, int /*c*/) {
   el->fCharEd.fKill.fMark = el->fLine.fCursor;
   return CC_NORM;
}


/* em_exchange_mark():
 *	Exchange the cursor and mark
 *	[^X^X]
 */
el_protected ElAction_t
/*ARGSUSED*/
em_exchange_mark(EditLine_t* el, int /*c*/) {
   char* cp;

   cp = el->fLine.fCursor;
   el->fLine.fCursor = el->fCharEd.fKill.fMark;
   el->fCharEd.fKill.fMark = cp;
   return CC_CURSOR;
}


/* em_universal_argument():
 *	Universal argument (argument times 4)
 *	[^U]
 */
el_protected ElAction_t
/*ARGSUSED*/
em_universal_argument(EditLine_t* el, int /*c*/) { /* multiply current argument by 4 */
   if (el->fState.fArgument > 1000000) {
      return CC_ERROR;
   }
   el->fState.fDoingArg = 1;
   el->fState.fArgument *= 4;
   return CC_ARGHACK;
}


/* em_meta_next():
 *	Add 8th bit to next character typed
 *	[<ESC>]
 */
el_protected ElAction_t
/*ARGSUSED*/
em_meta_next(EditLine_t* el, int /*c*/) {
   el->fState.fMetaNext = 1;
   return CC_ARGHACK;
}


/* em_toggle_overwrite():
 *	Switch from insert to overwrite mode or vice versa
 */
el_protected ElAction_t
/*ARGSUSED*/
em_toggle_overwrite(EditLine_t* el, int /*c*/) {
   el->fState.fInputMode = (el->fState.fInputMode == MODE_INSERT) ?
                            MODE_REPLACE : MODE_INSERT;
   return CC_NORM;
}


/* em_copy_prev_word():
 *	Copy current word to cursor
 */
el_protected ElAction_t
/*ARGSUSED*/
em_copy_prev_word(EditLine_t* el, int /*c*/) {
   char* cp, * oldc, * dp;

   if (el->fLine.fCursor == el->fLine.fBuffer) {
      return CC_ERROR;
   }

   oldc = el->fLine.fCursor;
   /* does a bounds check */
   cp = c__prev_word(el->fLine.fCursor, el->fLine.fBuffer,
                     el->fState.fArgument, ce__isword);

   c_insert(el, oldc - cp);

   for (dp = oldc; cp < oldc && dp < el->fLine.fLastChar; cp++) {
      *dp++ = *cp;
   }

   el->fLine.fCursor = dp;     /* put cursor at end */

   return CC_REFRESH;
} // em_copy_prev_word


/* em_inc_search_next():
 *	Emacs incremental next search
 */
el_protected ElAction_t
/*ARGSUSED*/
em_inc_search_next(EditLine_t* el, int /*c*/) {
   el->fSearch.fPatLen = 0;
   return ce_inc_search(el, ED_SEARCH_NEXT_HISTORY);
}


/* em_inc_search_prev():
 *	Emacs incremental reverse search
 */
el_protected ElAction_t
/*ARGSUSED*/
em_inc_search_prev(EditLine_t* el, int /*c*/) {
   el->fSearch.fPatLen = 0;
   return ce_inc_search(el, ED_SEARCH_PREV_HISTORY);
}


/* vi_undo():
 *	Vi undo last change
 *	[u]
 */
el_protected ElAction_t
/*ARGSUSED*/
em_undo(EditLine_t* el, int /*c*/) {
   char* cp, * kp;
   char temp;
   int i, size;
   CUndo_t* un = &el->fCharEd.fUndo;

#ifdef DEBUG_UNDO
      (void) fprintf(el->fErrFile, "Undo: %x \"%s\" +%d -%d\n",
                     un->fAction, un->fBuf, un->fISize, un->fDSize);
#endif

   switch (un->fAction) {
   case DELETE:

      if (un->fDSize == 0) {
         return CC_NORM;
      }

      (void) memcpy(un->fBuf, un->fPtr, un->fDSize);

      for (cp = un->fPtr; cp <= el->fLine.fLastChar; cp++) {
         *cp = cp[un->fDSize];
      }

      el->fLine.fLastChar -= un->fDSize;
      el->fLine.fCursor = un->fPtr;

      un->fAction = INSERT;
      un->fISize = un->fDSize;
      un->fDSize = 0;
      break;

   case DELETE | INSERT:
      size = un->fISize - un->fDSize;

      if (size > 0) {
         i = un->fDSize;
      } else {
         i = un->fISize;
      }
      cp = un->fPtr;
      kp = un->fBuf;

      while (i-- > 0) {
         temp = *kp;
         *kp++ = *cp;
         *cp++ = temp;
      }

      if (size > 0) {
         el->fLine.fCursor = cp;
         c_insert(el, size);

         while (size-- > 0 && cp < el->fLine.fLastChar) {
            temp = *kp;
            *kp++ = *cp;
            *cp++ = temp;
         }
      } else if (size < 0) {
         size = -size;

         for ( ; cp <= el->fLine.fLastChar; cp++) {
            *kp++ = *cp;
            *cp = cp[size];
         }
         el->fLine.fLastChar -= size;
      }
      el->fLine.fCursor = un->fPtr;
      i = un->fDSize;
      un->fDSize = un->fISize;
      un->fISize = i;
      break;

   case INSERT:

      if (un->fISize == 0) {
         return CC_NORM;
      }

      el->fLine.fCursor = un->fPtr;
      c_insert(el, (int) un->fISize);
      (void) memcpy(un->fPtr, un->fBuf, un->fISize);
      un->fAction = DELETE;
      un->fDSize = un->fISize;
      un->fISize = 0;
      break;

   case CHANGE:

      if (un->fISize == 0) {
         return CC_NORM;
      }

      el->fLine.fCursor = un->fPtr;
      size = (int) (el->fLine.fCursor - el->fLine.fLastChar);

      if ((unsigned int) size < un->fISize) {
         size = un->fISize;
      }
      cp = un->fPtr;
      kp = un->fBuf;

      for (i = 0; i < size; i++) {
         temp = *kp;
         *kp++ = *cp;
         *cp++ = temp;
      }
      un->fDSize = 0;
      break;

   default:
      return CC_ERROR;
   } // switch

   return CC_REFRESH;
} // em_undo
