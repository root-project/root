// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: chared.c,v 1.14 2001/05/17 01:02:17 christos Exp $	*/

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
 * chared.c: Character editor utilities
 */
#include "sys.h"

#include <stdlib.h>
#include "el.h"

/* value to leave unused in line buffer */
#define EL_LEAVE 2

/* cv_undo():
 *	Handle state for the vi undo command
 */
el_protected void
cv_undo(EditLine_t* el, int action, size_t size, char* ptr) {
   CUndo_t* vu = &el->fCharEd.fUndo;
   vu->fAction = action;
   vu->fPtr = ptr;
   vu->fISize = size;
   (void) memcpy(vu->fBuf, vu->fPtr, size);
#ifdef DEBUG_UNDO
      (void) fprintf(el->fErrFile, "Undo buffer \"%s\" size = +%d -%d\n",
                     vu->fPtr, vu->fISize, vu->fDSize);
#endif
}


/* c_insert():
 *	Insert num characters
 */
el_protected void
c_insert(EditLine_t* el, int num) {
   char* cp;

   if (el->fLine.fLastChar + num >= el->fLine.fLimit) {
      return;                           /* can't go past end of buffer */

   }

   if (el->fLine.fCursor < el->fLine.fLastChar) {
      /* if I must move chars */
      for (cp = el->fLine.fLastChar; cp >= el->fLine.fCursor; cp--) {
         cp[num] = *cp;
      }

      // work out equivalent offsets for colour buffer
      int colCursor = el->fLine.fCursor - el->fLine.fBuffer;
      int colLastChar = el->fLine.fLastChar - el->fLine.fBuffer;

      // shift colour buffer values along to match newly shifted positions
      for (int i = colLastChar; i >= colCursor; i--) {
         el->fLine.fBufColor[i + num] = el->fLine.fBufColor[i];
      }

   }
   el->fLine.fLastChar += num;
} // c_insert


/* c_delafter():
 *	Delete num characters after the cursor
 */
el_protected void
c_delafter(EditLine_t* el, int num) {
   if (el->fLine.fCursor + num > el->fLine.fLastChar) {
      num = el->fLine.fLastChar - el->fLine.fCursor;
   }

   if (num > 0) {
      char* cp;

      if (el->fMap.fCurrent != el->fMap.fEmacs) {
         cv_undo(el, INSERT, (size_t) num, el->fLine.fCursor);
      }

      for (cp = el->fLine.fCursor; cp <= el->fLine.fLastChar; cp++) {
         *cp = cp[num];
      }

      el->fLine.fLastChar -= num;
   }
} // c_delafter


/* c_delbefore():
 *	Delete num characters before the cursor
 */
el_protected void
c_delbefore(EditLine_t* el, int num) {
   if (el->fLine.fCursor - num < el->fLine.fBuffer) {
      num = el->fLine.fCursor - el->fLine.fBuffer;
   }

   if (num > 0) {
      char* cp;

      if (el->fMap.fCurrent != el->fMap.fEmacs) {
         cv_undo(el, INSERT, (size_t) num,
                 el->fLine.fCursor - num);
      }

      for (cp = el->fLine.fCursor - num;
           cp <= el->fLine.fLastChar;
           cp++) {
         *cp = cp[num];
      }

      el->fLine.fLastChar -= num;
   }
} // c_delbefore


/* ce__isword():
 *	Return if p is part of a word according to emacs
 */
el_protected int
ce__isword(int p) {
   return isalpha(p) || isdigit(p) || strchr("*?_-.[]~=", p) != NULL;
}


/* cv__isword():
 *	Return if p is part of a word according to vi
 */
el_protected int
cv__isword(int p) {
   return !isspace(p);
}


/* c__prev_word():
 *	Find the previous word
 */
el_protected char*
c__prev_word(char* p, char* low, int n, int (* wtest)(int)) {
   p--;

   while (n--) {
      while ((p >= low) && !(*wtest)((unsigned char) *p))
         p--;

      while ((p >= low) && (*wtest)((unsigned char) *p))
         p--;
   }

   /* cp now points to one character before the word */
   p++;

   if (p < low) {
      p = low;
   }
   /* cp now points where we want it */
   return p;
} // c__prev_word


/* c__next_word():
 *	Find the next word
 */
el_protected char*
c__next_word(char* p, char* high, int n, int (* wtest)(int)) {
   while (n--) {
      while ((p < high) && !(*wtest)((unsigned char) *p))
         p++;

      while ((p < high) && (*wtest)((unsigned char) *p))
         p++;
   }

   if (p > high) {
      p = high;
   }
   /* p now points where we want it */
   return p;
}


/* cv_next_word():
 *	Find the next word vi style
 */
el_protected char*
cv_next_word(EditLine_t* el, char* p, char* high, int n, int (* wtest)(int)) {
   int test;

   while (n--) {
      test = (*wtest)((unsigned char) *p);

      while ((p < high) && (*wtest)((unsigned char) *p) == test)
         p++;

      /*
       * vi historically deletes with cw only the word preserving the
       * trailing whitespace! This is not what 'w' does..
       */
      if (el->fCharEd.fVCmd.fAction != (DELETE | INSERT)) {
         while ((p < high) && isspace((unsigned char) *p))
            p++;
      }
   }

   /* p now points where we want it */
   if (p > high) {
      return high;
   } else {
      return p;
   }
} // cv_next_word


/* cv_prev_word():
 *	Find the previous word vi style
 */
el_protected char*
cv_prev_word(EditLine_t* el, char* p, char* low, int n, int (* wtest)(int)) {
   int test;

   while (n--) {
      p--;

      /*
       * vi historically deletes with cb only the word preserving the
       * leading whitespace! This is not what 'b' does..
       */
      if (el->fCharEd.fVCmd.fAction != (DELETE | INSERT)) {
         while ((p > low) && isspace((unsigned char) *p))
            p--;
      }
      test = (*wtest)((unsigned char) *p);

      while ((p >= low) && (*wtest)((unsigned char) *p) == test)
         p--;
      p++;

      while (isspace((unsigned char) *p))
         p++;
   }

   /* p now points where we want it */
   if (p < low) {
      return low;
   } else {
      return p;
   }
} // cv_prev_word


#ifdef notdef

/* c__number():
 *	Ignore character p points to, return number appearing after that.
 *      A '$' by itself means a big number; "$-" is for negative; '^' means 1.
 *      Return p pointing to last char used.
 */
el_protected char*
c__number(
   char* p,     /* character position */
   int* num,    /* Return value	*/
   int dval) {  /* dval is the number to subtract from like $-3 */
   int i;
   int sign = 1;

   if (*++p == '^') {
      *num = 1;
      return p;
   }

   if (*p == '$') {
      if (*++p != '-') {
         *num = 0x7fffffff;                     /* Handle $ */
         return --p;
      }
      sign = -1;                                /* Handle $- */
      ++p;
   }

   for (i = 0; isdigit((unsigned char) *p); i = 10 * i + *p++ - '0') {
      continue;
   }
   *num = (sign < 0 ? dval - i : i);
   return --p;
} // c__number


#endif

/* cv_delfini():
 *	Finish vi delete action
 */
el_protected void
cv_delfini(EditLine_t* el) {
   int size;
   int oaction;

   if (el->fCharEd.fVCmd.fAction & INSERT) {
      el->fMap.fCurrent = el->fMap.fKey;
   }

   oaction = el->fCharEd.fVCmd.fAction;
   el->fCharEd.fVCmd.fAction = NOP;

   if (el->fCharEd.fVCmd.fPos == 0) {
      return;
   }

   if (el->fLine.fCursor > el->fCharEd.fVCmd.fPos) {
      size = (int) (el->fLine.fCursor - el->fCharEd.fVCmd.fPos);
      c_delbefore(el, size);
      el->fLine.fCursor = el->fCharEd.fVCmd.fPos;
      re_refresh_cursor(el);
   } else if (el->fLine.fCursor < el->fCharEd.fVCmd.fPos) {
      size = (int) (el->fCharEd.fVCmd.fPos - el->fLine.fCursor);
      c_delafter(el, size);
   } else {
      size = 1;
      c_delafter(el, size);
   }

   switch (oaction) {
   case DELETE | INSERT:
      el->fCharEd.fUndo.fAction = DELETE | INSERT;
      break;
   case DELETE:
      el->fCharEd.fUndo.fAction = INSERT;
      break;
   case NOP:
   case INSERT:
   default:
      EL_ABORT((el->fErrFile, "Bad oaction %d\n", oaction));
      break;
   }


   el->fCharEd.fUndo.fPtr = el->fLine.fCursor;
   el->fCharEd.fUndo.fDSize = size;
} // cv_delfini


#ifdef notdef

/* ce__endword():
 *	Go to the end of this word according to emacs
 */
el_protected char*
ce__endword(char* p, char* high, int n) {
   p++;

   while (n--) {
      while ((p < high) && isspace((unsigned char) *p))
         p++;

      while ((p < high) && !isspace((unsigned char) *p))
         p++;
   }

   p--;
   return p;
}


#endif


/* cv__endword():
 *	Go to the end of this word according to vi
 */
el_protected char*
cv__endword(char* p, char* high, int n) {
   p++;

   while (n--) {
      while ((p < high) && isspace((unsigned char) *p))
         p++;

      if (isalnum((unsigned char) *p)) {
         while ((p < high) && isalnum((unsigned char) *p))
            p++;
      } else {
         while ((p < high) && !(isspace((unsigned char) *p) ||
                                isalnum((unsigned char) *p)))
            p++;
      }
   }
   p--;
   return p;
} // cv__endword


/* ch_init():
 *	Initialize the character editor
 */
el_protected int
ch_init(EditLine_t* el) {
   el->fLine.fBuffer = (char*) el_malloc(EL_BUFSIZ);
   el->fLine.fBufColor = (ElColor_t*) el_malloc(EL_BUFSIZ * sizeof(ElColor_t));

   if (el->fLine.fBuffer == NULL) {
      return -1;
   }

   (void) memset(el->fLine.fBuffer, 0, EL_BUFSIZ);
   (void) memset(el->fLine.fBufColor, 0, EL_BUFSIZ * sizeof(ElColor_t));
   el->fLine.fCursor = el->fLine.fBuffer;
   el->fLine.fLastChar = el->fLine.fBuffer;
   el->fLine.fLimit = &el->fLine.fBuffer[EL_BUFSIZ - 2];

   el->fCharEd.fUndo.fBuf = (char*) el_malloc(EL_BUFSIZ);

   if (el->fCharEd.fUndo.fBuf == NULL) {
      return -1;
   }
   (void) memset(el->fCharEd.fUndo.fBuf, 0, EL_BUFSIZ);
   el->fCharEd.fUndo.fAction = NOP;
   el->fCharEd.fUndo.fISize = 0;
   el->fCharEd.fUndo.fDSize = 0;
   el->fCharEd.fUndo.fPtr = el->fLine.fBuffer;

   el->fCharEd.fVCmd.fAction = NOP;
   el->fCharEd.fVCmd.fPos = el->fLine.fBuffer;
   el->fCharEd.fVCmd.fIns = el->fLine.fBuffer;

   el->fCharEd.fKill.fBuf = (char*) el_malloc(EL_BUFSIZ);

   if (el->fCharEd.fKill.fBuf == NULL) {
      return -1;
   }
   (void) memset(el->fCharEd.fKill.fBuf, 0, EL_BUFSIZ);
   el->fCharEd.fKill.fMark = el->fLine.fBuffer;
   el->fCharEd.fKill.fLast = el->fCharEd.fKill.fBuf;

   el->fMap.fCurrent = el->fMap.fKey;

   el->fState.fInputMode = MODE_INSERT;               /* XXX: save a default */
   el->fState.fDoingArg = 0;
   el->fState.fMetaNext = 0;
   el->fState.fArgument = 1;
   el->fState.fReplayHist = -1;
   el->fState.fLastCmd = ED_UNASSIGNED;

   el->fCharEd.fMacro.fNLine = NULL;
   el->fCharEd.fMacro.fLevel = -1;
   el->fCharEd.fMacro.fMacro = (char**) el_malloc(EL_MAXMACRO *
                                                    sizeof(char*));

   if (el->fCharEd.fMacro.fMacro == NULL) {
      return -1;
   }
   return 0;
} // ch_init


/* ch_reset():
 *	Reset the character editor
 */
el_protected void
ch_reset(EditLine_t* el) {
   el->fLine.fCursor = el->fLine.fBuffer;
   el->fLine.fLastChar = el->fLine.fBuffer;

   el->fCharEd.fUndo.fAction = NOP;
   el->fCharEd.fUndo.fISize = 0;
   el->fCharEd.fUndo.fDSize = 0;
   el->fCharEd.fUndo.fPtr = el->fLine.fBuffer;

   el->fCharEd.fVCmd.fAction = NOP;
   el->fCharEd.fVCmd.fPos = el->fLine.fBuffer;
   el->fCharEd.fVCmd.fIns = el->fLine.fBuffer;

   el->fCharEd.fKill.fMark = el->fLine.fBuffer;

   el->fMap.fCurrent = el->fMap.fKey;

   el->fState.fInputMode = MODE_INSERT;               /* XXX: save a default */
   el->fState.fDoingArg = 0;
   el->fState.fMetaNext = 0;
   el->fState.fArgument = 1;
   el->fState.fLastCmd = ED_UNASSIGNED;

   el->fCharEd.fMacro.fLevel = -1;

   el->fHistory.fEventNo = 0;
} // ch_reset


/* ch_enlargebufs():
 *	Enlarge line buffer to be able to hold twice as much characters.
 *	Also enlarge character colour buffer so that colour buffer is always the same size as the line buffer.
 *	Returns 1 if successful, 0 if not.
 */
el_protected int
ch_enlargebufs(EditLine_t* el, size_t addlen) {
   size_t sz, newsz;
   char* newbuffer, * oldbuf, * oldkbuf;
   ElColor_t* newcolorbuf, * oldcolorbuf;

   sz = el->fLine.fLimit - el->fLine.fBuffer + EL_LEAVE;
   newsz = sz * 2;

   /*
    * If newly required length is longer than current buffer, we need
    * to make the buffer big enough to hold both old and new stuff.
    */
   if (addlen > sz) {
      while (newsz - sz < addlen)
         newsz *= 2;
   }

   /*
    * Reallocate line buffer.
    */
   newbuffer = (char*) el_realloc(el->fLine.fBuffer, newsz);
   if (!newbuffer) {
      return 0;
   }

   newcolorbuf = (ElColor_t*) el_realloc(el->fLine.fBufColor, newsz * sizeof(ElColor_t));
   if (!newcolorbuf) {
      el_free((ptr_t) newbuffer);
      return 0;
   }


   /* zero the newly added memory, leave old data in */
   (void) memset(&newbuffer[sz], 0, newsz - sz);
   (void) memset(&newcolorbuf[sz], 0, newsz - sz);

   oldbuf = el->fLine.fBuffer;
   oldcolorbuf = el->fLine.fBufColor;

   el->fLine.fBuffer = newbuffer;
   el->fLine.fBufColor = newcolorbuf;
   el->fLine.fCursor = newbuffer + (el->fLine.fCursor - oldbuf);
   el->fLine.fLastChar = newbuffer + (el->fLine.fLastChar - oldbuf);
   el->fLine.fLimit = &newbuffer[newsz - EL_LEAVE];

// !!!!!!!!!!!!!!!!LOUISE: GOT AS FAR AS HERE !!!!!!!!!!!!!!!!!!!

   /*
    * Reallocate kill buffer.
    */
   newbuffer = (char*) el_realloc(el->fCharEd.fKill.fBuf, newsz);

   if (!newbuffer) {
      return 0;
   }

   /* zero the newly added memory, leave old data in */
   (void) memset(&newbuffer[sz], 0, newsz - sz);

   // LOUISE  - haven't changed the following 4 lines to reflect addition of colour buffer - may need to!!
   oldkbuf = el->fCharEd.fKill.fBuf;

   el->fCharEd.fKill.fBuf = newbuffer;
   el->fCharEd.fKill.fLast = newbuffer +
                               (el->fCharEd.fKill.fLast - oldkbuf);
   el->fCharEd.fKill.fMark = el->fLine.fBuffer +
                               (el->fCharEd.fKill.fMark - oldbuf);

   /*
    * Reallocate undo buffer.
    */
   newbuffer = (char*) el_realloc(el->fCharEd.fUndo.fBuf, newsz);

   if (!newbuffer) {
      return 0;
   }

   /* zero the newly added memory, leave old data in */
   (void) memset(&newbuffer[sz], 0, newsz - sz);

   el->fCharEd.fUndo.fPtr = el->fLine.fBuffer +
                              (el->fCharEd.fUndo.fPtr - oldbuf);
   el->fCharEd.fUndo.fBuf = newbuffer;

   if (!hist_enlargebuf(el, sz, newsz)) {
      return 0;
   }

   return 1;
} // ch_enlargebufs


/* ch_end():
 *	Free the data structures used by the editor
 */
el_protected void
ch_end(EditLine_t* el) {
   el_free((ptr_t) el->fLine.fBuffer);
   el->fLine.fBuffer = NULL;
   el_free((ptr_t) el->fLine.fBufColor);
   el->fLine.fBufColor = NULL;
   el->fLine.fLimit = NULL;
   el_free((ptr_t) el->fCharEd.fUndo.fBuf);
   el->fCharEd.fUndo.fBuf = NULL;
   el_free((ptr_t) el->fCharEd.fKill.fBuf);
   el->fCharEd.fKill.fBuf = NULL;
   el_free((ptr_t) el->fCharEd.fMacro.fMacro);
   el->fCharEd.fMacro.fMacro = NULL;
   ch_reset(el);
}


/* el_insertstr():
 *	Insert string at cursorI
 */
el_public int
el_insertstr(EditLine_t* el, const char* s) {
   size_t len;

   if ((len = strlen(s)) == 0) {
      return -1;
   }

   if (el->fLine.fLastChar + len >= el->fLine.fLimit) {
      if (!ch_enlargebufs(el, len)) {
         return -1;
      }
   }

   c_insert(el, (int) len);

   while (*s) {
      // set the colour information for the new character to default
      el->fLine.fBufColor[el->fLine.fCursor - el->fLine.fBuffer] = -1;
      // add the new character into el_line.fBuffer
      *el->fLine.fCursor++ = *s++;
   }
   return 0;
} // el_insertstr


/* el_deletestr():
 *	Delete num characters before the cursor
 */
el_public void
el_deletestr(EditLine_t* el, int n) {
   if (n <= 0) {
      return;
   }

   if (el->fLine.fCursor < &el->fLine.fBuffer[n]) {
      return;
   }

   c_delbefore(el, n);                  /* delete before dot */
   el->fLine.fCursor -= n;

   if (el->fLine.fCursor < el->fLine.fBuffer) {
      el->fLine.fCursor = el->fLine.fBuffer;
   }
}


/* c_gets():
 *	Get a string
 */
el_protected int
c_gets(EditLine_t* el, char* buf) {
   char ch;
   int len = 0;

   for (ch = 0; ch == 0;) {
      if (el_getc(el, &ch) != 1) {
         return ed_end_of_file(el, 0);
      }

      switch (ch) {
      case 0010:                /* Delete and backspace */
      case 0177:

         if (len > 1) {
            *el->fLine.fCursor-- = '\0';
            el->fLine.fLastChar = el->fLine.fCursor;
            buf[len--] = '\0';
         } else {
            el->fLine.fBuffer[0] = '\0';
            el->fLine.fLastChar = el->fLine.fBuffer;
            el->fLine.fCursor = el->fLine.fBuffer;
            return CC_REFRESH;
         }
         re_refresh(el);
         ch = 0;
         break;

      case 0033:                /* ESC */
      case '\r':                /* Newline */
      case '\n':
         break;

      default:

         if (len >= EL_BUFSIZ) {
            term_beep(el);
         } else {
            buf[len++] = ch;
            // set the colour information for the new character to default
            el->fLine.fBufColor[el->fLine.fCursor - el->fLine.fBuffer] = -1;
            // add the new character into el_line.fBuffer
            *el->fLine.fCursor++ = ch;
            el->fLine.fLastChar = el->fLine.fCursor;
         }
         re_refresh(el);
         ch = 0;
         break;
      } // switch
   }
   buf[len] = ch;
   return len;
} // c_gets


/* c_hpos():
 *	Return the current horizontal position of the cursor
 */
el_protected int
c_hpos(EditLine_t* el) {
   char* ptr;

   /*
    * Find how many characters till the beginning of this line.
    */
   if (el->fLine.fCursor == el->fLine.fBuffer) {
      return 0;
   } else {
      for (ptr = el->fLine.fCursor - 1;
           ptr >= el->fLine.fBuffer && *ptr != '\n';
           ptr--) {
         continue;
      }
      return el->fLine.fCursor - ptr - 1;
   }
}
