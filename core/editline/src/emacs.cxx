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
el_protected el_action_t
/*ARGSUSED*/
em_delete_or_list(EditLine* el, int /*c*/) {
   if (el->el_line.cursor == el->el_line.lastchar) {
      /* if I'm at the end */
      if (el->el_line.cursor == el->el_line.buffer) {
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
      c_delafter(el, el->el_state.argument);            /* delete after dot */

      if (el->el_line.cursor > el->el_line.lastchar) {
         el->el_line.cursor = el->el_line.lastchar;
      }
      /* bounds check */
      return CC_REFRESH;
   }
} // em_delete_or_list


/* em_delete_next_word():
 *	Cut from cursor to end of current word
 *	[M-d]
 */
el_protected el_action_t
/*ARGSUSED*/
em_delete_next_word(EditLine* el, int /*c*/) {
   char* cp, * p, * kp;

   if (el->el_line.cursor == el->el_line.lastchar) {
      return CC_ERROR;
   }

   cp = c__next_word(el->el_line.cursor, el->el_line.lastchar,
                     el->el_state.argument, ce__isword);

   for (p = el->el_line.cursor, kp = el->el_chared.c_kill.buf; p < cp; p++) {
      /* save the text */
      *kp++ = *p;
   }
   el->el_chared.c_kill.last = kp;

   c_delafter(el, cp - el->el_line.cursor);             /* delete after dot */

   if (el->el_line.cursor > el->el_line.lastchar) {
      el->el_line.cursor = el->el_line.lastchar;
   }
   /* bounds check */
   return CC_REFRESH;
} // em_delete_next_word


/* em_yank():
 *	Paste cut buffer at cursor position
 *	[^Y]
 */
el_protected el_action_t
/*ARGSUSED*/
em_yank(EditLine* el, int /*c*/) {
   char* kp, * cp;

   if (el->el_chared.c_kill.last == el->el_chared.c_kill.buf) {
      if (!ch_enlargebufs(el, 1)) {
         return CC_ERROR;
      }
   }

   if (el->el_line.lastchar +
       (el->el_chared.c_kill.last - el->el_chared.c_kill.buf) >=
       el->el_line.limit) {
      return CC_ERROR;
   }

   el->el_chared.c_kill.mark = el->el_line.cursor;
   cp = el->el_line.cursor;

   /* open the space, */
   c_insert(el, el->el_chared.c_kill.last - el->el_chared.c_kill.buf);

   /* copy the chars */
   for (kp = el->el_chared.c_kill.buf; kp < el->el_chared.c_kill.last; kp++) {
      *cp++ = *kp;
   }

   /* if an arg, cursor at beginning else cursor at end */
   if (el->el_state.argument == 1) {
      el->el_line.cursor = cp;
   }

   return CC_REFRESH;
} // em_yank


/* em_kill_line():
 *	Cut the entire line and save in cut buffer
 *	[^U]
 */
el_protected el_action_t
/*ARGSUSED*/
em_kill_line(EditLine* el, int /*c*/) {
   char* kp, * cp;

   cp = el->el_line.buffer;
   kp = el->el_chared.c_kill.buf;

   while (cp < el->el_line.lastchar)
      *kp++ = *cp++;            /* copy it */
   el->el_chared.c_kill.last = kp;
   /* zap! -- delete all of it */
   el->el_line.lastchar = el->el_line.buffer;
   el->el_line.cursor = el->el_line.buffer;
   return CC_REFRESH;
}


/* em_kill_region():
 *	Cut area between mark and cursor and save in cut buffer
 *	[^W]
 */
el_protected el_action_t
/*ARGSUSED*/
em_kill_region(EditLine* el, int /*c*/) {
   char* kp, * cp;

   if (!el->el_chared.c_kill.mark) {
      return CC_ERROR;
   }

   if (el->el_chared.c_kill.mark > el->el_line.cursor) {
      cp = el->el_line.cursor;
      kp = el->el_chared.c_kill.buf;

      while (cp < el->el_chared.c_kill.mark)
         *kp++ = *cp++;                 /* copy it */
      el->el_chared.c_kill.last = kp;
      c_delafter(el, cp - el->el_line.cursor);
   } else {                     /* mark is before cursor */
      cp = el->el_chared.c_kill.mark;
      kp = el->el_chared.c_kill.buf;

      while (cp < el->el_line.cursor)
         *kp++ = *cp++;                 /* copy it */
      el->el_chared.c_kill.last = kp;
      c_delbefore(el, cp - el->el_chared.c_kill.mark);
      el->el_line.cursor = el->el_chared.c_kill.mark;
   }
   return CC_REFRESH;
} // em_kill_region


/* em_copy_region():
 *	Copy area between mark and cursor to cut buffer
 *	[M-W]
 */
el_protected el_action_t
/*ARGSUSED*/
em_copy_region(EditLine* el, int /*c*/) {
   char* kp, * cp;

   if (el->el_chared.c_kill.mark) {
      return CC_ERROR;
   }

   if (el->el_chared.c_kill.mark > el->el_line.cursor) {
      cp = el->el_line.cursor;
      kp = el->el_chared.c_kill.buf;

      while (cp < el->el_chared.c_kill.mark)
         *kp++ = *cp++;                 /* copy it */
      el->el_chared.c_kill.last = kp;
   } else {
      cp = el->el_chared.c_kill.mark;
      kp = el->el_chared.c_kill.buf;

      while (cp < el->el_line.cursor)
         *kp++ = *cp++;                 /* copy it */
      el->el_chared.c_kill.last = kp;
   }
   return CC_NORM;
} // em_copy_region


/* em_gosmacs_traspose():
 *	Exchange the two characters before the cursor
 *	Gosling emacs transpose chars [^T]
 */
el_protected el_action_t
em_gosmacs_traspose(EditLine* el, int c) {
   if (el->el_line.cursor > &el->el_line.buffer[1]) {
      /* must have at least two chars entered */
      c = el->el_line.cursor[-2];
      el->el_line.cursor[-2] = el->el_line.cursor[-1];
      el->el_line.cursor[-1] = c;
      return CC_REFRESH;
   } else {
      return CC_ERROR;
   }
}


/* em_next_word():
 *	Move next to end of current word
 *	[M-f]
 */
el_protected el_action_t
/*ARGSUSED*/
em_next_word(EditLine* el, int /*c*/) {
   if (el->el_line.cursor == el->el_line.lastchar) {
      return CC_ERROR;
   }

   el->el_line.cursor = c__next_word(el->el_line.cursor,
                                     el->el_line.lastchar,
                                     el->el_state.argument,
                                     ce__isword);

   if (el->el_map.type == MAP_VI) {
      if (el->el_chared.c_vcmd.action & DELETE) {
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
el_protected el_action_t
/*ARGSUSED*/
em_upper_case(EditLine* el, int /*c*/) {
   char* cp, * ep;

   ep = c__next_word(el->el_line.cursor, el->el_line.lastchar,
                     el->el_state.argument, ce__isword);

   for (cp = el->el_line.cursor; cp < ep; cp++) {
      if (islower((unsigned char) *cp)) {
         *cp = toupper(*cp);
      }
   }

   el->el_line.cursor = ep;

   if (el->el_line.cursor > el->el_line.lastchar) {
      el->el_line.cursor = el->el_line.lastchar;
   }
   return CC_REFRESH;
} // em_upper_case


/* em_capitol_case():
 *	Capitalize the characters from cursor to end of current word
 *	[M-c]
 */
el_protected el_action_t
/*ARGSUSED*/
em_capitol_case(EditLine* el, int /*c*/) {
   char* cp, * ep;

   ep = c__next_word(el->el_line.cursor, el->el_line.lastchar,
                     el->el_state.argument, ce__isword);

   for (cp = el->el_line.cursor; cp < ep; cp++) {
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

   el->el_line.cursor = ep;

   if (el->el_line.cursor > el->el_line.lastchar) {
      el->el_line.cursor = el->el_line.lastchar;
   }
   return CC_REFRESH;
} // em_capitol_case


/* em_lower_case():
 *	Lowercase the characters from cursor to end of current word
 *	[M-l]
 */
el_protected el_action_t
/*ARGSUSED*/
em_lower_case(EditLine* el, int /*c*/) {
   char* cp, * ep;

   ep = c__next_word(el->el_line.cursor, el->el_line.lastchar,
                     el->el_state.argument, ce__isword);

   for (cp = el->el_line.cursor; cp < ep; cp++) {
      if (isupper((unsigned char) *cp)) {
         *cp = tolower(*cp);
      }
   }

   el->el_line.cursor = ep;

   if (el->el_line.cursor > el->el_line.lastchar) {
      el->el_line.cursor = el->el_line.lastchar;
   }
   return CC_REFRESH;
} // em_lower_case


/* em_set_mark():
 *	Set the mark at cursor
 *	[^@]
 */
el_protected el_action_t
/*ARGSUSED*/
em_set_mark(EditLine* el, int /*c*/) {
   el->el_chared.c_kill.mark = el->el_line.cursor;
   return CC_NORM;
}


/* em_exchange_mark():
 *	Exchange the cursor and mark
 *	[^X^X]
 */
el_protected el_action_t
/*ARGSUSED*/
em_exchange_mark(EditLine* el, int /*c*/) {
   char* cp;

   cp = el->el_line.cursor;
   el->el_line.cursor = el->el_chared.c_kill.mark;
   el->el_chared.c_kill.mark = cp;
   return CC_CURSOR;
}


/* em_universal_argument():
 *	Universal argument (argument times 4)
 *	[^U]
 */
el_protected el_action_t
/*ARGSUSED*/
em_universal_argument(EditLine* el, int /*c*/) { /* multiply current argument by 4 */
   if (el->el_state.argument > 1000000) {
      return CC_ERROR;
   }
   el->el_state.doingarg = 1;
   el->el_state.argument *= 4;
   return CC_ARGHACK;
}


/* em_meta_next():
 *	Add 8th bit to next character typed
 *	[<ESC>]
 */
el_protected el_action_t
/*ARGSUSED*/
em_meta_next(EditLine* el, int /*c*/) {
   el->el_state.metanext = 1;
   return CC_ARGHACK;
}


/* em_toggle_overwrite():
 *	Switch from insert to overwrite mode or vice versa
 */
el_protected el_action_t
/*ARGSUSED*/
em_toggle_overwrite(EditLine* el, int /*c*/) {
   el->el_state.inputmode = (el->el_state.inputmode == MODE_INSERT) ?
                            MODE_REPLACE : MODE_INSERT;
   return CC_NORM;
}


/* em_copy_prev_word():
 *	Copy current word to cursor
 */
el_protected el_action_t
/*ARGSUSED*/
em_copy_prev_word(EditLine* el, int /*c*/) {
   char* cp, * oldc, * dp;

   if (el->el_line.cursor == el->el_line.buffer) {
      return CC_ERROR;
   }

   oldc = el->el_line.cursor;
   /* does a bounds check */
   cp = c__prev_word(el->el_line.cursor, el->el_line.buffer,
                     el->el_state.argument, ce__isword);

   c_insert(el, oldc - cp);

   for (dp = oldc; cp < oldc && dp < el->el_line.lastchar; cp++) {
      *dp++ = *cp;
   }

   el->el_line.cursor = dp;     /* put cursor at end */

   return CC_REFRESH;
} // em_copy_prev_word


/* em_inc_search_next():
 *	Emacs incremental next search
 */
el_protected el_action_t
/*ARGSUSED*/
em_inc_search_next(EditLine* el, int /*c*/) {
   el->el_search.patlen = 0;
   return ce_inc_search(el, ED_SEARCH_NEXT_HISTORY);
}


/* em_inc_search_prev():
 *	Emacs incremental reverse search
 */
el_protected el_action_t
/*ARGSUSED*/
em_inc_search_prev(EditLine* el, int /*c*/) {
   el->el_search.patlen = 0;
   return ce_inc_search(el, ED_SEARCH_PREV_HISTORY);
}


/* vi_undo():
 *	Vi undo last change
 *	[u]
 */
el_protected el_action_t
/*ARGSUSED*/
em_undo(EditLine* el, int /*c*/) {
   char* cp, * kp;
   char temp;
   int i, size;
   c_undo_t* un = &el->el_chared.c_undo;

#ifdef DEBUG_UNDO
      (void) fprintf(el->el_errfile, "Undo: %x \"%s\" +%d -%d\n",
                     un->action, un->buf, un->isize, un->dsize);
#endif

   switch (un->action) {
   case DELETE:

      if (un->dsize == 0) {
         return CC_NORM;
      }

      (void) memcpy(un->buf, un->ptr, un->dsize);

      for (cp = un->ptr; cp <= el->el_line.lastchar; cp++) {
         *cp = cp[un->dsize];
      }

      el->el_line.lastchar -= un->dsize;
      el->el_line.cursor = un->ptr;

      un->action = INSERT;
      un->isize = un->dsize;
      un->dsize = 0;
      break;

   case DELETE | INSERT:
      size = un->isize - un->dsize;

      if (size > 0) {
         i = un->dsize;
      } else {
         i = un->isize;
      }
      cp = un->ptr;
      kp = un->buf;

      while (i-- > 0) {
         temp = *kp;
         *kp++ = *cp;
         *cp++ = temp;
      }

      if (size > 0) {
         el->el_line.cursor = cp;
         c_insert(el, size);

         while (size-- > 0 && cp < el->el_line.lastchar) {
            temp = *kp;
            *kp++ = *cp;
            *cp++ = temp;
         }
      } else if (size < 0) {
         size = -size;

         for ( ; cp <= el->el_line.lastchar; cp++) {
            *kp++ = *cp;
            *cp = cp[size];
         }
         el->el_line.lastchar -= size;
      }
      el->el_line.cursor = un->ptr;
      i = un->dsize;
      un->dsize = un->isize;
      un->isize = i;
      break;

   case INSERT:

      if (un->isize == 0) {
         return CC_NORM;
      }

      el->el_line.cursor = un->ptr;
      c_insert(el, (int) un->isize);
      (void) memcpy(un->ptr, un->buf, un->isize);
      un->action = DELETE;
      un->dsize = un->isize;
      un->isize = 0;
      break;

   case CHANGE:

      if (un->isize == 0) {
         return CC_NORM;
      }

      el->el_line.cursor = un->ptr;
      size = (int) (el->el_line.cursor - el->el_line.lastchar);

      if ((unsigned int) size < un->isize) {
         size = un->isize;
      }
      cp = un->ptr;
      kp = un->buf;

      for (i = 0; i < size; i++) {
         temp = *kp;
         *kp++ = *cp;
         *cp++ = temp;
      }
      un->dsize = 0;
      break;

   default:
      return CC_ERROR;
   } // switch

   return CC_REFRESH;
} // em_undo
