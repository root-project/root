// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: common.c,v 1.10 2001/01/10 07:45:41 jdolecek Exp $	*/

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
 * common.c: Common Editor functions
 */
#include "sys.h"
#include "el.h"

/* ed_end_of_file():
 *	Indicate end of file
 *	[^D]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_end_of_file(EditLine_t* el, int /*c*/) {
   re_goto_bottom(el);
   *el->fLine.fLastChar = '\0';
   return CC_EOF;
}


/* ed_insert():
 *	Add character to the line
 *	Insert a character [bound to all insert keys]
 */
el_protected ElAction_t
ed_insert(EditLine_t* el, int c) {
   int i;

   if (c == '\0') {
      return CC_ERROR;
   }

   if (el->fLine.fLastChar + el->fState.fArgument >=
       el->fLine.fLimit) {
      /* end of buffer space, try to allocate more */
      if (!ch_enlargebufs(el, (size_t) el->fState.fArgument)) {
         return CC_ERROR;                       /* error allocating more */
      }
   }

   if (el->fState.fArgument == 1) {
      if (el->fState.fInputMode != MODE_INSERT) {
         el->fCharEd.fUndo.fBuf[el->fCharEd.fUndo.fISize++] =
            *el->fLine.fCursor;
         el->fCharEd.fUndo.fBuf[el->fCharEd.fUndo.fISize] =
            '\0';
         c_delafter(el, 1);
      }
      c_insert(el, 1);

      // set the colour information for the new character to default
      el->fLine.fBufColor[el->fLine.fCursor - el->fLine.fBuffer] = -1;
      // add the new character into el_line.fBuffer
      *el->fLine.fCursor++ = c;

      el->fState.fDoingArg = 0;                /* just in case */
      re_fastaddc(el);                          /* fast refresh for one char. */

   } else {
      if (el->fState.fInputMode != MODE_INSERT) {
         for (i = 0; i < el->fState.fArgument; i++) {
            el->fCharEd.fUndo.fBuf[el->fCharEd.fUndo.fISize++] =
               el->fLine.fCursor[i];
         }

         el->fCharEd.fUndo.fBuf[el->fCharEd.fUndo.fISize] =
            '\0';
         c_delafter(el, el->fState.fArgument);
      }
      c_insert(el, el->fState.fArgument);

      while (el->fState.fArgument--) {
         // set the colour information for the new character to default
         el->fLine.fBufColor[el->fLine.fCursor - el->fLine.fBuffer] = -1;
         // add the new character into el_line.fBuffer
         *el->fLine.fCursor++ = c;
      }
      re_refresh(el);
   }

   /*
      if (el->fState.fInputMode == MODE_REPLACE_1)
           (void) vi_command_mode(el, 0);
    */

   return CC_NORM;
} // ed_insert


/* ed_delete_prev_word():
 *	Delete from beginning of current word to cursor
 *	[M-^?] [^W]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_delete_prev_word(EditLine_t* el, int /*c*/) {
   char* cp, * p, * kp;

   if (el->fLine.fCursor == el->fLine.fBuffer) {
      return CC_ERROR;
   }

   cp = c__prev_word(el->fLine.fCursor, el->fLine.fBuffer,
                     el->fState.fArgument, ce__isword);

   for (p = cp, kp = el->fCharEd.fKill.fBuf; p < el->fLine.fCursor; p++) {
      *kp++ = *p;
   }
   el->fCharEd.fKill.fLast = kp;

   c_delbefore(el, el->fLine.fCursor - cp);            /* delete before dot */
   el->fLine.fCursor = cp;

   if (el->fLine.fCursor < el->fLine.fBuffer) {
      el->fLine.fCursor = el->fLine.fBuffer;           /* bounds check */
   }
   return CC_REFRESH;
} // ed_delete_prev_word


/* ed_delete_next_char():
 *	Delete character under cursor
 *	[^D] [x]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_delete_next_char(EditLine_t* el, int /*c*/) {
#ifdef notdef                   /* XXX */
# define EL el->fLine
      (void) fprintf(el->el_errlfile,
                     "\nD(b: %x(%s)  c: %x(%s) last: %x(%s) limit: %x(%s)\n",
                     EL.fBuffer, EL.fBuffer, EL.fCursor, EL.fCursor, EL.fLastChar,
                     EL.fLastChar, EL.fLimit, EL.fLimit);
#endif

   if (el->fLine.fCursor == el->fLine.fLastChar) {
      /* if I'm at the end */
      if (el->fMap.fType == MAP_VI) {
         if (el->fLine.fCursor == el->fLine.fBuffer) {
            /* if I'm also at the beginning */
#ifdef KSHVI
            return CC_ERROR;
#else
            term_overwrite(el, STReof, 0, 4);
            /* then do a EOF */
            term__flush();
            return CC_EOF;
#endif
         } else {
#ifdef KSHVI
            el->fLine.fCursor--;
#else
            return CC_ERROR;
#endif
         }
      } else {
         if (el->fLine.fCursor != el->fLine.fBuffer) {
            el->fLine.fCursor--;
         } else {
            return CC_ERROR;
         }
      }
   }
   c_delafter(el, el->fState.fArgument);       /* delete after dot */

   if (el->fLine.fCursor >= el->fLine.fLastChar &&
       el->fLine.fCursor > el->fLine.fBuffer) {
      /* bounds check */
      el->fLine.fCursor = el->fLine.fLastChar - 1;
   }
   return CC_REFRESH;
} // ed_delete_next_char


/* ed_kill_line():
 *	Cut to the end of line
 *	[^K] [^K]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_kill_line(EditLine_t* el, int /*c*/) {
   char* kp, * cp;

   cp = el->fLine.fCursor;
   kp = el->fCharEd.fKill.fBuf;

   while (cp < el->fLine.fLastChar)
      *kp++ = *cp++;            /* copy it */
   el->fCharEd.fKill.fLast = kp;
   /* zap! -- delete to end */
   el->fLine.fLastChar = el->fLine.fCursor;
   return CC_REFRESH;
}


/* ed_move_to_end():
 *	Move cursor to the end of line
 *	[^E] [^E]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_move_to_end(EditLine_t* el, int /*c*/) {
   el->fLine.fCursor = el->fLine.fLastChar;

   if (el->fMap.fType == MAP_VI) {
#ifdef VI_MOVE
      el->fLine.fCursor--;
#endif

      if (el->fCharEd.fVCmd.fAction & DELETE) {
         cv_delfini(el);
         return CC_REFRESH;
      }
   }
   return CC_CURSOR;
}


/* ed_move_to_beg():
 *	Move cursor to the beginning of line
 *	[^A] [^A]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_move_to_beg(EditLine_t* el, int /*c*/) {
   el->fLine.fCursor = el->fLine.fBuffer;

   if (el->fMap.fType == MAP_VI) {
      /* We want FIRST non space character */
      while (isspace((unsigned char) *el->fLine.fCursor))
         el->fLine.fCursor++;

      if (el->fCharEd.fVCmd.fAction & DELETE) {
         cv_delfini(el);
         return CC_REFRESH;
      }
   }
   return CC_CURSOR;
}


/* ed_transpose_chars():
 *	Exchange the character to the left of the cursor with the one under it
 *	[^T] [^T]
 */
el_protected ElAction_t
ed_transpose_chars(EditLine_t* el, int c) {
   if (el->fLine.fCursor < el->fLine.fLastChar) {
      if (el->fLine.fLastChar <= &el->fLine.fBuffer[1]) {
         return CC_ERROR;
      } else {
         el->fLine.fCursor++;
      }
   }

   if (el->fLine.fCursor > &el->fLine.fBuffer[1]) {
      /* must have at least two chars entered */
      c = el->fLine.fCursor[-2];
      el->fLine.fCursor[-2] = el->fLine.fCursor[-1];
      el->fLine.fCursor[-1] = c;
      return CC_REFRESH;
   } else {
      return CC_ERROR;
   }
} // ed_transpose_chars


/* ed_next_char():
 *	Move to the right one character
 *	[^F] [^F]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_next_char(EditLine_t* el, int /*c*/) {
   if (el->fLine.fCursor >= el->fLine.fLastChar) {
      return CC_ERROR;
   }

   el->fLine.fCursor += el->fState.fArgument;

   if (el->fLine.fCursor > el->fLine.fLastChar) {
      el->fLine.fCursor = el->fLine.fLastChar;
   }

   if (el->fMap.fType == MAP_VI) {
      if (el->fCharEd.fVCmd.fAction & DELETE) {
         cv_delfini(el);
         return CC_REFRESH;
      }
   }
   return CC_CURSOR;
} // ed_next_char


/* ed_prev_word():
 *	Move to the beginning of the current word
 *	[M-b] [b]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_prev_word(EditLine_t* el, int /*c*/) {
   if (el->fLine.fCursor == el->fLine.fBuffer) {
      return CC_ERROR;
   }

   el->fLine.fCursor = c__prev_word(el->fLine.fCursor,
                                     el->fLine.fBuffer,
                                     el->fState.fArgument,
                                     ce__isword);

   if (el->fMap.fType == MAP_VI) {
      if (el->fCharEd.fVCmd.fAction & DELETE) {
         cv_delfini(el);
         return CC_REFRESH;
      }
   }
   return CC_CURSOR;
} // ed_prev_word


/* ed_prev_char():
 *	Move to the left one character
 *	[^B] [^B]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_prev_char(EditLine_t* el, int /*c*/) {
   if (el->fLine.fCursor > el->fLine.fBuffer) {
      el->fLine.fCursor -= el->fState.fArgument;

      if (el->fLine.fCursor < el->fLine.fBuffer) {
         el->fLine.fCursor = el->fLine.fBuffer;
      }

      if (el->fMap.fType == MAP_VI) {
         if (el->fCharEd.fVCmd.fAction & DELETE) {
            cv_delfini(el);
            return CC_REFRESH;
         }
      }
      return CC_CURSOR;
   } else {
      return CC_ERROR;
   }
} // ed_prev_char


/* ed_quoted_insert():
 *	Add the next character typed verbatim
 *	[^V] [^V]
 */
el_protected ElAction_t
ed_quoted_insert(EditLine_t* el, int c) {
   int num;
   char tc;

   tty_quotemode(el);
   num = el_getc(el, &tc);
   c = (unsigned char) tc;
   tty_noquotemode(el);

   if (num == 1) {
      return ed_insert(el, c);
   } else {
      return ed_end_of_file(el, 0);
   }
}


/* ed_digit():
 *	Adds to argument or enters a digit
 */
el_protected ElAction_t
ed_digit(EditLine_t* el, int c) {
   if (!isdigit(c)) {
      return CC_ERROR;
   }

   if (el->fState.fDoingArg) {
      /* if doing an arg, add this in... */
      if (el->fState.fLastCmd == EM_UNIVERSAL_ARGUMENT) {
         el->fState.fArgument = c - '0';
      } else {
         if (el->fState.fArgument > 1000000) {
            return CC_ERROR;
         }
         el->fState.fArgument =
            (el->fState.fArgument * 10) + (c - '0');
      }
      return CC_ARGHACK;
   } else {
      if (el->fLine.fLastChar + 1 >= el->fLine.fLimit) {
         if (!ch_enlargebufs(el, 1)) {
            return CC_ERROR;
         }
      }

      if (el->fState.fInputMode != MODE_INSERT) {
         el->fCharEd.fUndo.fBuf[el->fCharEd.fUndo.fISize++] =
            *el->fLine.fCursor;
         el->fCharEd.fUndo.fBuf[el->fCharEd.fUndo.fISize] =
            '\0';
         c_delafter(el, 1);
      }
      c_insert(el, 1);

      // set the colour information for the new character to default
      el->fLine.fBufColor[el->fLine.fCursor - el->fLine.fBuffer] = -1;
      // add the new character into el_line.fBuffer
      *el->fLine.fCursor++ = c;

      el->fState.fDoingArg = 0;
      re_fastaddc(el);
   }
   return CC_NORM;
} // ed_digit


/* ed_argument_digit():
 *	Digit that starts argument
 *	For ESC-n
 */
el_protected ElAction_t
ed_argument_digit(EditLine_t* el, int c) {
   if (!isdigit(c)) {
      return CC_ERROR;
   }

   if (el->fState.fDoingArg) {
      if (el->fState.fArgument > 1000000) {
         return CC_ERROR;
      }
      el->fState.fArgument = (el->fState.fArgument * 10) +
                              (c - '0');
   } else {                     /* else starting an argument */
      el->fState.fArgument = c - '0';
      el->fState.fDoingArg = 1;
   }
   return CC_ARGHACK;
} // ed_argument_digit


/* ed_unassigned():
 *	Indicates unbound character
 *	Bound to keys that are not assigned
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_unassigned(EditLine_t* el, int /*c*/) {
   term_beep(el);
   term__flush();
   return CC_NORM;
}


/**
** TTY key handling.
**/

/* ed_tty_sigint():
 *	Tty interrupt character
 *	[^C]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_tty_sigint(EditLine_t* /*el*/, int /*c*/) {
   return CC_NORM;
}


/* ed_tty_dsusp():
 *	Tty delayed suspend character
 *	[^Y]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_tty_dsusp(EditLine_t* /*el*/, int /*c*/) {
   return CC_NORM;
}


/* ed_tty_flush_output():
 *	Tty flush output characters
 *	[^O]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_tty_flush_output(EditLine_t* /*el*/, int /*c*/) {
   return CC_NORM;
}


/* ed_tty_sigquit():
 *	Tty quit character
 *	[^\]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_tty_sigquit(EditLine_t* /*el*/, int /*c*/) {
   return CC_NORM;
}


/* ed_tty_sigtstp():
 *	Tty suspend character
 *	[^Z]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_tty_sigtstp(EditLine_t* /*el*/, int /*c*/) {
   return CC_NORM;
}


/* ed_tty_stop_output():
 *	Tty disallow output characters
 *	[^S]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_tty_stop_output(EditLine_t* /*el*/, int /*c*/) {
   return CC_NORM;
}


/* ed_tty_start_output():
 *	Tty allow output characters
 *	[^Q]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_tty_start_output(EditLine_t* /*el*/, int /*c*/) {
   return CC_NORM;
}


/* ed_newline():
 *	Execute command
 *	[^J]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_newline(EditLine_t* el, int /*c*/) {
   re_goto_bottom(el);
   // *el->fLine.fLastChar++ = '\n';
   // ^^^ cmt'd out by stephan, because:
   // a) it's lame. nobody expects to get the \n back.
   // b) the above code doesn't KNOW if lastchar is valid, and doesn't check.
   // c) See (a).
   *el->fLine.fLastChar = '\0';

   if (el->fMap.fType == MAP_VI) {
      el->fCharEd.fVCmd.fIns = el->fLine.fBuffer;
   }
   return CC_NEWLINE;
}


/* ed_delete_prev_char():
 *	Delete the character to the left of the cursor
 *	[^?]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_delete_prev_char(EditLine_t* el, int /*c*/) {
   if (el->fLine.fCursor <= el->fLine.fBuffer) {
      return CC_ERROR;
   }

   c_delbefore(el, el->fState.fArgument);
   el->fLine.fCursor -= el->fState.fArgument;

   if (el->fLine.fCursor < el->fLine.fBuffer) {
      el->fLine.fCursor = el->fLine.fBuffer;
   }
   return CC_REFRESH;
}


/* ed_clear_screen():
 *	Clear screen leaving current line at the top
 *	[^L]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_clear_screen(EditLine_t* el, int /*c*/) {
   term_clear_screen(el);       /* clear the whole real screen */
   re_clear_display(el);        /* reset everything */
   return CC_REFRESH;
}


/* ed_redisplay():
 *	Redisplay everything
 *	^R
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_redisplay(EditLine_t* /*el*/, int /*c*/) {
   return CC_REDISPLAY;
}


/* ed_start_over():
 *	Erase current line and start from scratch
 *	[^G]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_start_over(EditLine_t* el, int /*c*/) {
   ch_reset(el);
   return CC_REFRESH;
}


/* ed_sequence_lead_in():
 *	First character in a bound sequence
 *	Placeholder for external keys
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_sequence_lead_in(EditLine_t* /*el*/, int /*c*/) {
   return CC_NORM;
}


/* ed_prev_history():
 *	Move to the previous history line
 *	[^P] [k]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_prev_history(EditLine_t* el, int /*c*/) {
   char beep = 0;

   el->fCharEd.fUndo.fAction = NOP;
   *el->fLine.fLastChar = '\0';                /* just in case */

   if (el->fHistory.fEventNo == 0) {           /* save the current buffer
                                                 * away */
      (void) strncpy(el->fHistory.fBuf, el->fLine.fBuffer,
                     EL_BUFSIZ);
      el->fHistory.fLast = el->fHistory.fBuf +
                            (el->fLine.fLastChar - el->fLine.fBuffer);
   }
   el->fHistory.fEventNo += el->fState.fArgument;

   if (hist_get(el) == CC_ERROR) {
      beep = 1;
      /* el->fHistory.fEventNo was fixed by first call */
      (void) hist_get(el);
   }
   re_refresh(el);

   if (beep) {
      return CC_ERROR;
   } else {
      return CC_NORM;                   /* was CC_UP_HIST */
   }
} // ed_prev_history


/* ed_next_history():
 *	Move to the next history line
 *	[^N] [j]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_next_history(EditLine_t* el, int /*c*/) {
   el->fCharEd.fUndo.fAction = NOP;
   *el->fLine.fLastChar = '\0';        /* just in case */

   if (el->fHistory.fEventNo == 0 && el->fState.fArgument == 1) {
      /* ROOT special treatment: kill the current buffer,
         it's used as workaround for ^C which is caught by CINT */
      el->fLine.fCursor = el->fLine.fBuffer;
      return ed_kill_line(el, 0);
   } else {
      el->fHistory.fEventNo -= el->fState.fArgument;

      if (el->fHistory.fEventNo < 0) {
         el->fHistory.fEventNo = 0;
         return CC_ERROR;            /* make it beep */
      }
   }
   return hist_get(el);
}


/* ed_search_prev_history():
 *	Search previous in history for a line matching the current
 *	next search history [M-P] [K]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_search_prev_history(EditLine_t* el, int /*c*/) {
   const char* hp;
   int h;
   ElBool_t found = 0;

   el->fCharEd.fVCmd.fAction = NOP;
   el->fCharEd.fUndo.fAction = NOP;
   *el->fLine.fLastChar = '\0';        /* just in case */

   if (el->fHistory.fEventNo < 0) {
#ifdef DEBUG_EDIT
         (void) fprintf(el->fErrFile,
                        "e_prev_search_hist(): eventno < 0;\n");
#endif
      el->fHistory.fEventNo = 0;
      return CC_ERROR;
   }

   if (el->fHistory.fEventNo == 0) {
      (void) strncpy(el->fHistory.fBuf, el->fLine.fBuffer,
                     EL_BUFSIZ);
      el->fHistory.fLast = el->fHistory.fBuf +
                            (el->fLine.fLastChar - el->fLine.fBuffer);
   }

   if (el->fHistory.fRef == NULL) {
      return CC_ERROR;
   }

   hp = HIST_FIRST(el);

   if (hp == NULL) {
      return CC_ERROR;
   }

   c_setpat(el);                /* Set search pattern !! */

   for (h = 1; h <= el->fHistory.fEventNo; h++) {
      hp = HIST_NEXT(el);
   }

   while (hp != NULL) {
#ifdef SDEBUG
         (void) fprintf(el->fErrFile, "Comparing with \"%s\"\n", hp);
#endif

      if ((strncmp(hp, el->fLine.fBuffer, (size_t)
                   (el->fLine.fLastChar - el->fLine.fBuffer)) ||
           hp[el->fLine.fLastChar - el->fLine.fBuffer]) &&
          c_hmatch(el, hp)) {
         found++;
         break;
      }
      h++;
      hp = HIST_NEXT(el);
   }

   if (!found) {
#ifdef SDEBUG
         (void) fprintf(el->fErrFile, "not found\n");
#endif
      return CC_ERROR;
   }
   el->fHistory.fEventNo = h;

   return hist_get(el);
} // ed_search_prev_history


/* ed_search_next_history():
 *	Search next in history for a line matching the current
 *	[M-N] [J]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_search_next_history(EditLine_t* el, int /*c*/) {
   const char* hp;
   int h;
   ElBool_t found = 0;

   el->fCharEd.fVCmd.fAction = NOP;
   el->fCharEd.fUndo.fAction = NOP;
   *el->fLine.fLastChar = '\0';        /* just in case */

   if (el->fHistory.fEventNo == 0) {
      return CC_ERROR;
   }

   if (el->fHistory.fRef == NULL) {
      return CC_ERROR;
   }

   hp = HIST_FIRST(el);

   if (hp == NULL) {
      return CC_ERROR;
   }

   c_setpat(el);                /* Set search pattern !! */

   for (h = 1; h < el->fHistory.fEventNo && hp; h++) {
#ifdef SDEBUG
         (void) fprintf(el->fErrFile, "Comparing with \"%s\"\n", hp);
#endif

      if ((strncmp(hp, el->fLine.fBuffer, (size_t)
                   (el->fLine.fLastChar - el->fLine.fBuffer)) ||
           hp[el->fLine.fLastChar - el->fLine.fBuffer]) &&
          c_hmatch(el, hp)) {
         found = h;
      }
      hp = HIST_NEXT(el);
   }

   if (!found) {                /* is it the current history number? */
      if (!c_hmatch(el, el->fHistory.fBuf)) {
#ifdef SDEBUG
            (void) fprintf(el->fErrFile, "not found\n");
#endif
         return CC_ERROR;
      }
   }
   el->fHistory.fEventNo = found;

   return hist_get(el);
} // ed_search_next_history


/* ed_prev_line():
 *	Move up one line
 *	Could be [k] [^p]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_prev_line(EditLine_t* el, int /*c*/) {
   char* ptr;
   int nchars = c_hpos(el);

   /*
    * Move to the line requested
    */
   if (*(ptr = el->fLine.fCursor) == '\n') {
      ptr--;
   }

   for ( ; ptr >= el->fLine.fBuffer; ptr--) {
      if (*ptr == '\n' && --el->fState.fArgument <= 0) {
         break;
      }
   }

   if (el->fState.fArgument > 0) {
      return CC_ERROR;
   }

   /*
    * Move to the beginning of the line
    */
   for (ptr--; ptr >= el->fLine.fBuffer && *ptr != '\n'; ptr--) {
      continue;
   }

   /*
    * Move to the character requested
    */
   for (ptr++;
        nchars-- > 0 && ptr < el->fLine.fLastChar && *ptr != '\n';
        ptr++) {
      continue;
   }

   el->fLine.fCursor = ptr;
   return CC_CURSOR;
} // ed_prev_line


/* ed_next_line():
 *	Move down one line
 *	Could be [j] [^n]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_next_line(EditLine_t* el, int /*c*/) {
   char* ptr;
   int nchars = c_hpos(el);

   /*
    * Move to the line requested
    */
   for (ptr = el->fLine.fCursor; ptr < el->fLine.fLastChar; ptr++) {
      if (*ptr == '\n' && --el->fState.fArgument <= 0) {
         break;
      }
   }

   if (el->fState.fArgument > 0) {
      return CC_ERROR;
   }

   /*
    * Move to the character requested
    */
   for (ptr++;
        nchars-- > 0 && ptr < el->fLine.fLastChar && *ptr != '\n';
        ptr++) {
      continue;
   }

   el->fLine.fCursor = ptr;
   return CC_CURSOR;
} // ed_next_line


/* ed_command():
 *	Editline extended command
 *	[M-X] [:]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_command(EditLine_t* el, int /*c*/) {
   char tmpbuf[EL_BUFSIZ];
   int tmplen;

   el->fLine.fBuffer[0] = '\0';
   el->fLine.fLastChar = el->fLine.fBuffer;
   el->fLine.fCursor = el->fLine.fBuffer;

   c_insert(el, 3);             /* prompt + ": " */

   // set the colour information for the new character to default
   el->fLine.fBufColor[el->fLine.fCursor - el->fLine.fBuffer] = -1;
   // add the new character into el_line.fBuffer
   *el->fLine.fCursor++ = '\n';

   // set the colour information for the new character to default
   el->fLine.fBufColor[el->fLine.fCursor - el->fLine.fBuffer] = -1;
   // add the new character into el_line.fBuffer
   *el->fLine.fCursor++ = ':';

   // set the colour information for the new character to default
   el->fLine.fBufColor[el->fLine.fCursor - el->fLine.fBuffer] = -1;
   // add the new character into el_line.fBuffer
   *el->fLine.fCursor++ = ' ';

   re_refresh(el);

   tmplen = c_gets(el, tmpbuf);
   tmpbuf[tmplen] = '\0';

   el->fLine.fBuffer[0] = '\0';
   el->fLine.fLastChar = el->fLine.fBuffer;
   el->fLine.fCursor = el->fLine.fBuffer;

   if (parse_line(el, tmpbuf) == -1) {
      return CC_ERROR;
   } else {
      return CC_REFRESH;
   }
} // ed_command


/* ed_replay_hist():
 *	Replay n-th history entry
 *	[^O]
 */
el_protected ElAction_t
/*ARGSUSED*/
ed_replay_hist(EditLine_t* el, int /*c*/) {
   static const char newline[] = "\n";
   // current history idx:
   if (el->fState.fReplayHist < 0) {
      // store the hist idx for repeated ^O
      el->fState.fReplayHist = el->fHistory.fEventNo - 1;
   }
   // execute the line as if the user pressed enter
   el_push(el, newline);

   // run whatever would be run if was entered
   return ed_newline(el, '\n');
}
