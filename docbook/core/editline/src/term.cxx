// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: term.c,v 1.32 2001/01/23 15:55:31 jdolecek Exp $	*/

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
 * term.c: Editor/termcap-curses interface
 *	   We have to declare a static variable here, since the
 *	   termcap putchar routine does not take an argument!
 */
#include "sys.h"
#include <stdio.h>
#include <signal.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <string>

#include "el.h"
#include "TTermManip.h"

// termcap.h an be extremely "dirty", polluting with CPP macros,
// so #include last!
#include "rlcurses.h"

/*
 * IMPORTANT NOTE: these routines are allowed to look at the current screen
 * and the current possition assuming that it is correct.  If this is not
 * true, then the update will be WRONG!  This is (should be) a valid
 * assumption...
 */

#define TC_BUFSIZE 2048

#define GoodStr(a) (el->fTerm.fStr[a] != NULL && \
                    el->fTerm.fStr[a][0] != '\0')
#define Str(a) el->fTerm.fStr[a]
#define Val(a) el->fTerm.fVal[a]

#ifdef notdef
el_private const struct {
   const char* b_name;
   int b_rate;
} baud_rate[] = {
# ifdef B0
   { "0", B0 },
# endif
# ifdef B50
   { "50", B50 },
# endif
# ifdef B75
   { "75", B75 },
# endif
# ifdef B110
   { "110", B110 },
# endif
# ifdef B134
   { "134", B134 },
# endif
# ifdef B150
   { "150", B150 },
# endif
# ifdef B200
   { "200", B200 },
# endif
# ifdef B300
   { "300", B300 },
# endif
# ifdef B600
   { "600", B600 },
# endif
# ifdef B900
   { "900", B900 },
# endif
# ifdef B1200
   { "1200", B1200 },
# endif
# ifdef B1800
   { "1800", B1800 },
# endif
# ifdef B2400
   { "2400", B2400 },
# endif
# ifdef B3600
   { "3600", B3600 },
# endif
# ifdef B4800
   { "4800", B4800 },
# endif
# ifdef B7200
   { "7200", B7200 },
# endif
# ifdef B9600
   { "9600", B9600 },
# endif
# ifdef EXTA
   { "19200", EXTA },
# endif
# ifdef B19200
   { "19200", B19200 },
# endif
# ifdef EXTB
   { "38400", EXTB },
# endif
# ifdef B38400
   { "38400", B38400 },
# endif
   { NULL, 0 }
};
#endif

el_private const struct TermCapStr_t {
   const char* fName;
   const char* fLongName;
} tstr[] = {
#define T_al 0
   { "al", "add new blank line" },
#define T_bl 1
   { "bl", "audible bell" },
#define T_cd 2
   { "cd", "clear to bottom" },
#define T_ce 3
   { "ce", "clear to end of line" },
#define T_ch 4
   { "ch", "cursor to horiz pos" },
#define T_cl 5
   { "cl", "clear screen" },
#define T_dc 6
   { "dc", "delete a character" },
#define T_dl 7
   { "dl", "delete a line" },
#define T_dm 8
   { "dm", "start delete mode" },
#define T_ed 9
   { "ed", "end delete mode" },
#define T_ei 10
   { "ei", "end insert mode" },
#define T_fs 11
   { "fs", "cursor from status line" },
#define T_ho 12
   { "ho", "home cursor" },
#define T_ic 13
   { "ic", "insert character" },
#define T_im 14
   { "im", "start insert mode" },
#define T_ip 15
   { "ip", "insert padding" },
#define T_kd 16
   { "kd", "sends cursor down" },
#define T_kl 17
   { "kl", "sends cursor left" },
#define T_kr 18
   { "kr", "sends cursor right" },
#define T_ku 19
   { "ku", "sends cursor up" },
#define T_md 20
   { "md", "begin bold" },
#define T_me 21
   { "me", "end attributes" },
#define T_nd 22
   { "nd", "non destructive space" },
#define T_se 23
   { "se", "end standout" },
#define T_so 24
   { "so", "begin standout" },
#define T_ts 25
   { "ts", "cursor to status line" },
#define T_up 26
   { "up", "cursor up one" },
#define T_us 27
   { "us", "begin underline" },
#define T_ue 28
   { "ue", "end underline" },
#define T_vb 29
   { "vb", "visible bell" },
#define T_DC 30
   { "DC", "delete multiple chars" },
#define T_DO 31
   { "DO", "cursor down multiple" },
#define T_IC 32
   { "IC", "insert multiple chars" },
#define T_LE 33
   { "LE", "cursor left multiple" },
#define T_RI 34
   { "RI", "cursor right multiple" },
#define T_UP 35
   { "UP", "cursor up multiple" },
#define T_kh 36
   { "kh", "send cursor home" },
#define T_at7 37
   { "@7", "send cursor end" },
#define T_kD 38
   { "kD", "delete a character" },
#define T_str 39
   { NULL, NULL }
};

el_private const struct TermCapVal_t {
   const char* fName;
   const char* fLongName;
} tval[] = {
#define T_am 0
   { "am", "has automatic margins" },
#define T_pt 1
   { "pt", "has physical tabs" },
#define T_li 2
   { "li", "Number of lines" },
#define T_co 3
   { "co", "Number of columns" },
#define T_km 4
   { "km", "Has meta key" },
#define T_xt 5
   { "xt", "Tab chars destructive" },
#define T_xn 6
   { "xn", "newline ignored at right margin" },
#define T_MT 7
   { "MT", "Has meta key" },                            /* XXX? */
#define T_val 8
   { NULL, NULL, }
};
/* do two or more of the attributes use me */

el_private void term_setflags(EditLine_t*);
el_private int term_rebuffer_display(EditLine_t*);
el_private void term_free_display(EditLine_t*);
el_private int term_alloc_display(EditLine_t*);
el_private void term_alloc(EditLine_t*, const struct TermCapStr_t*, const char*);
el_private void term_init_arrow(EditLine_t*);
el_private void term_reset_arrow(EditLine_t*);
el_private void term_init_color(EditLine_t*);


el_private FILE* term_outfile = NULL;   /* XXX: How do we fix that? */


/* term_setflags():
 *	Set the terminal capability flags
 */
el_private void
term_setflags(EditLine_t* el) {
   EL_FLAGS = 0;

   if (el->fTTY.t_tabs) {
      EL_FLAGS |= (Val(T_pt) && !Val(T_xt)) ? TERM_CAN_TAB : 0;
   }

   EL_FLAGS |= (Val(T_km) || Val(T_MT)) ? TERM_HAS_META : 0;
   EL_FLAGS |= GoodStr(T_ce) ? TERM_CAN_CEOL : 0;
   EL_FLAGS |= (GoodStr(T_dc) || GoodStr(T_DC)) ? TERM_CAN_DELETE : 0;
   EL_FLAGS |= (GoodStr(T_im) || GoodStr(T_ic) || GoodStr(T_IC)) ?
               TERM_CAN_INSERT : 0;
   EL_FLAGS |= (GoodStr(T_up) || GoodStr(T_UP)) ? TERM_CAN_UP : 0;
   EL_FLAGS |= Val(T_am) ? TERM_HAS_AUTO_MARGINS : 0;
   EL_FLAGS |= Val(T_xn) ? TERM_HAS_MAGIC_MARGINS : 0;

   if (GoodStr(T_me) && GoodStr(T_ue)) {
      EL_FLAGS |= (strcmp(Str(T_me), Str(T_ue)) == 0) ?
                  TERM_CAN_ME : 0;
   } else {
      EL_FLAGS &= ~TERM_CAN_ME;
   }

   if (GoodStr(T_me) && GoodStr(T_se)) {
      EL_FLAGS |= (strcmp(Str(T_me), Str(T_se)) == 0) ?
                  TERM_CAN_ME : 0;
   }


#ifdef DEBUG_SCREEN

   if (!EL_CAN_UP) {
      (void) fprintf(el->fErrFile,
                     "WARNING: Your terminal cannot move up.\n");
      (void) fprintf(el->fErrFile,
                     "Editing may be odd for long lines.\n");
   }

   if (!EL_CAN_CEOL) {
      (void) fprintf(el->fErrFile, "no clear EOL capability.\n");
   }

   if (!EL_CAN_DELETE) {
      (void) fprintf(el->fErrFile, "no delete char capability.\n");
   }

   if (!EL_CAN_INSERT) {
      (void) fprintf(el->fErrFile, "no insert char capability.\n");
   }
#endif /* DEBUG_SCREEN */
} // term_setflags


/* term_init():
 *	Initialize the terminal stuff
 */
el_protected int
term_init(EditLine_t* el) {
   el->fTerm.fBuf = (char*) el_malloc(TC_BUFSIZE);

   if (el->fTerm.fBuf == NULL) {
      return -1;
   }
   el->fTerm.fCap = (char*) el_malloc(TC_BUFSIZE);

   if (el->fTerm.fCap == NULL) {
      return -1;
   }
   el->fTerm.fFKey = (FKey_t*) el_malloc(A_K_NKEYS * sizeof(FKey_t));

   if (el->fTerm.fFKey == NULL) {
      return -1;
   }
   el->fTerm.fLoc = 0;
   el->fTerm.fStr = (char**) el_malloc(T_str * sizeof(char*));

   if (el->fTerm.fStr == NULL) {
      return -1;
   }
   (void) memset(el->fTerm.fStr, 0, T_str * sizeof(char*));
   el->fTerm.fVal = (int*) el_malloc(T_val * sizeof(int));

   if (el->fTerm.fVal == NULL) {
      return -1;
   }
   (void) memset(el->fTerm.fVal, 0, T_val * sizeof(int));
   term_outfile = el->fOutFile;

   if (term_set(el, NULL) == -1) {
      return -1;
   }
   term_init_arrow(el);

   term_init_color(el);

   return 0;
} // term_init


/* term_end():
 *	Clean up the terminal stuff
 */
el_protected void
term_end(EditLine_t* el) {
   el_free((ptr_t) el->fTerm.fBuf);
   el->fTerm.fBuf = NULL;
   el_free((ptr_t) el->fTerm.fCap);
   el->fTerm.fCap = NULL;
   el->fTerm.fLoc = 0;
   el_free((ptr_t) el->fTerm.fStr);
   el->fTerm.fStr = NULL;
   el_free((ptr_t) el->fTerm.fVal);
   el->fTerm.fVal = NULL;
   term_free_display(el);
}


/* term_alloc():
 *	Maintain a string pool for termcap strings
 */
el_private void
term_alloc(EditLine_t* el, const struct TermCapStr_t* t, const char* cap) {
   char termbuf[TC_BUFSIZE];
   int tlen, clen;
   char** tlist = el->fTerm.fStr;
   char** tmp, ** str = &tlist[t - tstr];

   if (cap == NULL || *cap == '\0') {
      *str = NULL;
      return;
   } else {
      clen = strlen(cap);
   }

   tlen = *str == NULL ? 0 : strlen(*str);

   /*
    * New string is shorter; no need to allocate space
    */
   if (clen <= tlen) {
      // coverity[secure_coding]
      (void) strcpy(*str, cap);                 /* XXX strcpy is safe */
      return;
   }

   /*
    * New string is longer; see if we have enough space to append
    */
   if (el->fTerm.fLoc + 3 < TC_BUFSIZE) {
      /* XXX strcpy is safe */
      // coverity[secure_coding]
      (void) strcpy(*str = &el->fTerm.fBuf[el->fTerm.fLoc],
                    cap);
      el->fTerm.fLoc += clen + 1;            /* one for \0 */
      return;
   }

   /*
    * Compact our buffer; no need to check compaction, cause we know it
    * fits...
    */
   tlen = 0;

   for (tmp = tlist; tmp < &tlist[T_str]; tmp++) {
      if (*tmp != NULL && *tmp != '\0' && *tmp != *str) {
         char* ptr;

         for (ptr = *tmp; *ptr != '\0'; termbuf[tlen++] = *ptr++) {
            continue;
         }
         termbuf[tlen++] = '\0';
      }
   }
   memcpy(el->fTerm.fBuf, termbuf, TC_BUFSIZE);
   el->fTerm.fLoc = tlen;

   if (el->fTerm.fLoc + 3 >= TC_BUFSIZE) {
      (void) fprintf(el->fErrFile,
                     "Out of termcap string space.\n");
      return;
   }
   /* XXX strcpy is safe */
   // coverity[secure_coding]
   (void) strcpy(*str = &el->fTerm.fBuf[el->fTerm.fLoc], cap);
   el->fTerm.fLoc += clen + 1;       /* one for \0 */
   return;
} // term_alloc


/* term_rebuffer_display():
 *	Rebuffer the display after the screen changed size
 */
el_private int
term_rebuffer_display(EditLine_t* el) {
   ElCoord_t* c = &el->fTerm.fSize;

   term_free_display(el);

   c->fH = Val(T_co);
   c->fV = Val(T_li);

   if (term_alloc_display(el) == -1) {
      return -1;
   }
   return 0;
}


/* term_alloc_display():
 *	Allocate a new display.
 */
el_private int
term_alloc_display(EditLine_t* el) {                      // LOUISE COLOUR : duplicated all display functionality for colour
   int i;
   char** b;
   ElColor_t** col;
   ElCoord_t* c = &el->fTerm.fSize;

   // original display
   b = (char**) el_malloc((size_t) (sizeof(char*) * (c->fV + 1)));

   if (b == NULL) {
      return -1;
   }

   for (i = 0; i < c->fV; i++) {
      b[i] = (char*) el_malloc((size_t) (sizeof(char) * (c->fH + 1)));

      if (b[i] == NULL) {
         el_free((ptr_t) b);
         return -1;
      }
   }
   b[c->fV] = NULL;
   el->fDisplay = b;

   // duplicate el_display for el_dispcolor
   col = (ElColor_t**) el_malloc((size_t) (sizeof(ElColor_t*) * (c->fV + 1)));

   if (col == NULL) {
      return -1;
   }

   for (i = 0; i < c->fV; i++) {
      col[i] = (ElColor_t*) el_malloc((size_t) (sizeof(ElColor_t) * (c->fH + 1)));

      if (col[i] == NULL) {
         el_free((ptr_t) col);
         return -1;
      }
   }
   col[c->fV] = NULL;
   el->fDispColor = col;

   // original el_vdisplay code
   b = (char**) el_malloc((size_t) (sizeof(char*) * (c->fV + 1)));

   if (b == NULL) {
      return -1;
   }

   for (i = 0; i < c->fV; i++) {
      b[i] = (char*) el_malloc((size_t) (sizeof(char) * (c->fH + 1)));

      if (b[i] == NULL) {
         for (int ii = 0; ii < i; ++ii) {
            el_free((ptr_t) b[ii]);
         }
         el_free((ptr_t) b);
         return -1;
      }
   }
   b[c->fV] = NULL;
   el->fVDisplay = b;

   // duplicate el_vdisplay functionality for el_vdispcolor
   col = (ElColor_t**) el_malloc((size_t) (sizeof(ElColor_t*) * (c->fV + 1)));

   if (col == NULL) {
      return -1;
   }

   for (i = 0; i < c->fV; i++) {
      col[i] = (ElColor_t*) el_malloc((size_t) (sizeof(ElColor_t) * (c->fH + 1)));

      if (col[i] == NULL) {
         for (int ii = 0; ii < i; ++ii) {
            el_free((ptr_t) col[ii]);
         }
         el_free((ptr_t) col);
         return -1;
      }
   }
   col[c->fV] = NULL;
   el->fVDispColor = col;

   return 0;
} // term_alloc_display


/* term_free_display():
 *	Free the display buffers
 */
el_private void
term_free_display(EditLine_t* el) {               // LOUISE COLOUR : duplicated all display functionality for colour
   char** b;
   char** bufp;
   ElColor_t** c;
   ElColor_t** bufc;

   // free display (original)
   b = el->fDisplay;
   el->fDisplay = NULL;

   if (b != NULL) {
      for (bufp = b; *bufp != NULL; bufp++) {
         el_free((ptr_t) *bufp);
      }
      el_free((ptr_t) b);
   }

   // free display colour info
   c = el->fDispColor;
   el->fDispColor = NULL;

   if (c != NULL) {
      for (bufc = c; *bufc != NULL; bufc++) {
         el_free((ptr_t) *bufc);
      }
      el_free((ptr_t) c);
   }

   // free vdisplay (original)
   b = el->fVDisplay;
   el->fVDisplay = NULL;

   if (b != NULL) {
      for (bufp = b; *bufp != NULL; bufp++) {
         el_free((ptr_t) *bufp);
      }
      el_free((ptr_t) b);
   }

   // free vdisplay colour info
   c = el->fVDispColor;
   el->fVDispColor = NULL;

   if (c != NULL) {
      for (bufc = c; *bufc != NULL; bufc++) {
         el_free((ptr_t) *bufc);
      }
      el_free((ptr_t) c);
   }
} // term_free_display


/* term_move_to_line():
 *	move to line <where> (first line == 0)
 *      as efficiently as possible
 */
el_protected void
term_move_to_line(EditLine_t* el, int where) {
   int del;

   if (where == el->fCursor.fV) {
      return;
   }

   if (where > el->fTerm.fSize.fV) {
#ifdef DEBUG_SCREEN
         (void) fprintf(el->fErrFile,
                        "term_move_to_line: where is ridiculous: %d\r\n", where);
#endif /* DEBUG_SCREEN */
      return;
   }

   if ((del = where - el->fCursor.fV) > 0) {
      while (del > 0) {
         if (EL_HAS_AUTO_MARGINS &&
             el->fDisplay[el->fCursor.fV][0] != '\0') {
            /* move without newline */
            term_move_to_char(el, el->fTerm.fSize.fH - 1);
            term_overwrite(el,
                           &el->fDisplay[el->fCursor.fV][el->fCursor.fH],
                           &el->fDispColor[el->fCursor.fV][el->fCursor.fH],
                           1);
            /* updates Cursor */
            del--;
         } else {
            if ((del > 1) && GoodStr(T_DO)) {
               (void) tputs(tgoto(Str(T_DO), del, del), del, term__putc);
               del = 0;
            } else {
               for ( ; del > 0; del--) {
                  term__putcolorch('\n', NULL);
               }
               /* because the \n will become \r\n */
               el->fCursor.fH = 0;
            }
         }
      }
   } else {                     /* del < 0 */
      if (GoodStr(T_UP) && (-del > 1 || !GoodStr(T_up))) {
         (void) tputs(tgoto(Str(T_UP), -del, -del), -del, term__putc);
      } else {
         if (GoodStr(T_up)) {
            for ( ; del < 0; del++) {
               (void) tputs(Str(T_up), 1, term__putc);
            }
         }
      }
   }
   el->fCursor.fV = where;     /* now where is here */
} // term_move_to_line


/* term_move_to_char():
 *	Move to the character position specified
 */
el_protected void
term_move_to_char(EditLine_t* el, int where) {
   int del, i;

mc_again:

   if (where == el->fCursor.fH) {
      return;
   }

   if (where > el->fTerm.fSize.fH) {
#ifdef DEBUG_SCREEN
         (void) fprintf(el->fErrFile,
                        "term_move_to_char: where is riduculous: %d\r\n", where);
#endif /* DEBUG_SCREEN */
      return;
   }

   if (!where) {                /* if where is first column */
      term__putcolorch('\r', NULL);             /* do a CR */
      el->fCursor.fH = 0;
      return;
   }
   del = where - el->fCursor.fH;

   if ((del < -4 || del > 4) && GoodStr(T_ch)) {
      /* go there directly */
      (void) tputs(tgoto(Str(T_ch), where, where), where, term__putc);
   } else {
      if (del > 0) {            /* moving forward */
         if ((del > 4) && GoodStr(T_RI)) {
            (void) tputs(tgoto(Str(T_RI), del, del), del, term__putc);
         } else {
            /* if I can do tabs, use them */
            if (EL_CAN_TAB) {
               if ((el->fCursor.fH & 0370) !=
                   (where & 0370)) {
                  /* if not within tab stop */
                  for (i =
                          (el->fCursor.fH & 0370);
                       i < (where & 0370);
                       i += 8) {
                     term__putcolorch('\t', NULL);
                  }
                  /* then tab over */
                  el->fCursor.fH = where & 0370;
               }
            }

            /*
             * it's usually cheaper to just write the
             * chars, so we do.
             */

            /*
             * NOTE THAT term_overwrite() WILL CHANGE
             * el->fCursor.fH!!!
             */
            term_overwrite(el,
                           &el->fDisplay[el->fCursor.fV][el->fCursor.fH],
                           &el->fDispColor[el->fCursor.fV][el->fCursor.fH],
                           where - el->fCursor.fH);

         }
      } else {                  /* del < 0 := moving backward */
         if ((-del > 4) && GoodStr(T_LE)) {
            (void) tputs(tgoto(Str(T_LE), -del, -del), -del, term__putc);
         } else {               /* can't go directly there */
                                /*
                                 * if the "cost" is greater than the "cost"
                                 * from col 0
                                 */
            if (EL_CAN_TAB ?
                (-del > ((where >> 3) +
                         (where & 07)))
                : (-del > where)) {
               term__putcolorch('\r', NULL);                            /* do a CR */
               el->fCursor.fH = 0;
               goto mc_again;                           /* and try again */
            }

            for (i = 0; i < -del; i++) {
               term__putcolorch('\b', &el->fDispColor[el->fCursor.fV][el->fCursor.fH]);
            }
         }
      }
   }
   el->fCursor.fH = where;                     /* now where is here */
} // term_move_to_char


/* term_overwrite():
 *	Overstrike num characters
 */
el_protected void
term_overwrite(EditLine_t* el, const char* cp, ElColor_t* color, int n) {
   if (n <= 0) {
      return;                   /* catch bugs */

   }

   if (n > el->fTerm.fSize.fH) {
#ifdef DEBUG_SCREEN
         (void) fprintf(el->fErrFile,
                        "term_overwrite: n is riduculous: %d\r\n", n);
#endif /* DEBUG_SCREEN */
      return;
   }

   do {
      if (color) {
         el->fDispColor[el->fCursor.fV][el->fCursor.fH] = *color;
      }
      term__putcolorch(*cp++, /*&(el->fLine.fBufColor[cp - el->fLine.fBuffer])*/ color ? color++ : 0);
      el->fCursor.fH++;
   }
   while (--n);

   if (el->fCursor.fH >= el->fTerm.fSize.fH) {       /* wrap? */
      if (EL_HAS_AUTO_MARGINS) {                /* yes */
         el->fCursor.fH = 0;
         el->fCursor.fV++;

         if (EL_HAS_MAGIC_MARGINS) {
            /* force the wrap to avoid the "magic"
             * situation */
            char c;

            if ((c = el->fDisplay[el->fCursor.fV][el->fCursor.fH]) != '\0') {
               term_overwrite(el, &c, &el->fDispColor[el->fCursor.fV][el->fCursor.fH], 1);
            } else {
               term__putcolorch(' ', NULL);
            }
            el->fCursor.fH = 1;
         }
      } else {                  /* no wrap, but cursor stays on screen */
         el->fCursor.fH = el->fTerm.fSize.fH;
      }
   }
} // term_overwrite


/* term_deletechars():
 *	Delete num characters
 */
el_protected void
term_deletechars(EditLine_t* el, int num) {
   if (num <= 0) {
      return;
   }

   if (!EL_CAN_DELETE) {
#ifdef DEBUG_EDIT
         (void) fprintf(el->fErrFile, "   ERROR: cannot delete   \n");
#endif /* DEBUG_EDIT */
      return;
   }

   if (num > el->fTerm.fSize.fH) {
#ifdef DEBUG_SCREEN
         (void) fprintf(el->fErrFile,
                        "term_deletechars: num is riduculous: %d\r\n", num);
#endif /* DEBUG_SCREEN */
      return;
   }

   if (GoodStr(T_DC)) {         /* if I have multiple delete */
      if ((num > 1) || !GoodStr(T_dc)) {                /* if dc would be more
                                                         * expen. */
         (void) tputs(tgoto(Str(T_DC), num, num), num, term__putc);
         return;
      }
   }

   if (GoodStr(T_dm)) {         /* if I have delete mode */
      (void) tputs(Str(T_dm), 1, term__putc);
   }

   if (GoodStr(T_dc)) {         /* else do one at a time */
      while (num--)
         (void) tputs(Str(T_dc), 1, term__putc);
   }

   if (GoodStr(T_ed)) {         /* if I have delete mode */
      (void) tputs(Str(T_ed), 1, term__putc);
   }
} // term_deletechars


/* term_insertwrite():
 *	Puts terminal in insert character mode or inserts num
 *	characters in the line
 */
el_protected void
term_insertwrite(EditLine_t* el, const char* cp, ElColor_t* color, int num) {
   if (num <= 0) {
      return;
   }

   if (!EL_CAN_INSERT) {
#ifdef DEBUG_EDIT
         (void) fprintf(el->fErrFile, "   ERROR: cannot insert   \n");
#endif /* DEBUG_EDIT */
      return;
   }

   if (num > el->fTerm.fSize.fH) {
#ifdef DEBUG_SCREEN
         (void) fprintf(el->fErrFile,
                        "StartInsert: num is riduculous: %d\r\n", num);
#endif /* DEBUG_SCREEN */
      return;
   }

   if (GoodStr(T_IC)) {         /* if I have multiple insert */
      if ((num > 1) || !GoodStr(T_ic)) {
         /* if ic would be more expensive */

         (void) tputs(tgoto(Str(T_IC), num, num), num, term__putc);
         term_overwrite(el, cp, color, num);

         /* this updates el_cursor.fH */
         return;
      }
   }

   if (GoodStr(T_im) && GoodStr(T_ei)) {        /* if I have insert mode */
      (void) tputs(Str(T_im), 1, term__putc);

      el->fCursor.fH += num;

      do {
         // need to get color info about cp
         term__putcolorch(*cp++, /*&(el->fLine.fBufColor[cp - el->fLine.fBuffer])*/ color ? color++ : 0);
      }
      while (--num);

      if (GoodStr(T_ip)) {              /* have to make num chars insert */
         (void) tputs(Str(T_ip), 1, term__putc);
      }

      (void) tputs(Str(T_ei), 1, term__putc);
      return;
   }

   do {
      if (GoodStr(T_ic)) {              /* have to make num chars insert */
         (void) tputs(Str(T_ic), 1, term__putc);
      }
      /* insert a char */
      term__putcolorch(*cp++, /*&(el->fLine.fBufColor[cp - el->fLine.fBuffer])*/ color ? color++ : 0);

      el->fCursor.fH++;

      if (GoodStr(T_ip)) {              /* have to make num chars insert */
         (void) tputs(Str(T_ip), 1, term__putc);
      }
      /* pad the inserted char */

   }
   while (--num);
} // term_insertwrite


/* term_clear_EOL():
 *	clear to end of line.  There are num characters to clear
 */
el_protected void
term_clear_EOL(EditLine_t* el, int num) {
   int i;

   if (EL_CAN_CEOL && GoodStr(T_ce)) {
      (void) tputs(Str(T_ce), 1, term__putc);
   } else {
      for (i = 0; i < num; i++) {
         term__putcolorch(' ', NULL);
      }
      el->fCursor.fH += num;           /* have written num spaces */
   }
}


/* term_clear_screen():
 *	Clear the screen
 */
el_protected void
term_clear_screen(EditLine_t* el) { /* clear the whole screen and home */
   if (GoodStr(T_cl)) {
      /* send the clear screen code */
      (void) tputs(Str(T_cl), Val(T_li), term__putc);
   } else if (GoodStr(T_ho) && GoodStr(T_cd)) {
      (void) tputs(Str(T_ho), Val(T_li), term__putc);           /* home */
      /* clear to bottom of screen */
      (void) tputs(Str(T_cd), Val(T_li), term__putc);
   } else {
      term__putcolorch('\r', NULL);
      term__putcolorch('\n', NULL);
   }
}


/* term_beep():
 *	Beep the way the terminal wants us
 */
el_protected void
term_beep(EditLine_t* el) {
   if (GoodStr(T_bl)) {
      /* what termcap says we should use */
      (void) tputs(Str(T_bl), 1, term__putc);
   } else {
      term__putcolorch('\007', NULL);           /* an ASCII bell; ^G */
   }
}


#ifdef notdef

/* term_clear_to_bottom():
 *	Clear to the bottom of the screen
 */
el_protected void
term_clear_to_bottom(EditLine_t* el) {
   if (GoodStr(T_cd)) {
      (void) tputs(Str(T_cd), Val(T_li), term__putc);
   } else if (GoodStr(T_ce)) {
      (void) tputs(Str(T_ce), Val(T_li), term__putc);
   }
}


#endif


/* term_set():
 *	Read in the terminal capabilities from the requested terminal
 */
el_protected int
term_set(EditLine_t* el, const char* term) {
   int i;
   char buf[TC_BUFSIZE];
   char* area;
   const struct TermCapStr_t* t;
   sigset_t oset, nset;
   int lins, cols;

   (void) sigemptyset(&nset);
   (void) sigaddset(&nset, SIGWINCH);
   (void) sigprocmask(SIG_BLOCK, &nset, &oset);

   area = buf;

   if (term == NULL) {
      term = getenv("TERM");
   }

   if (!term || !term[0]
       || !isatty(0)
       || !isatty(1)) {
      term = "dumb";
   }

   if (strcmp(term, "emacs") == 0
       || !isatty(0)) {
      el->fFlags |= EDIT_DISABLED;
   }

   memset(el->fTerm.fCap, 0, TC_BUFSIZE);

   i = tgetent(el->fTerm.fCap, term);

   if (i <= 0) {
      if (i == -1) {
         (void) fprintf(el->fErrFile,
                        "Cannot read termcap database;\n");
      } else if (i == 0) {
         (void) fprintf(el->fErrFile,
                        "No entry for terminal type \"%s\";\n", term);
      }
      (void) fprintf(el->fErrFile,
                     "using dumb terminal settings.\n");
      Val(T_co) = 80;           /* do a dumb terminal */
      Val(T_pt) = Val(T_km) = Val(T_li) = 0;
      Val(T_xt) = Val(T_MT);

      for (t = tstr; t->fName != NULL; t++) {
         term_alloc(el, t, NULL);
      }
   } else {
      /* auto/magic margins */
      Val(T_am) = tgetflag((char*)"am");
      Val(T_xn) = tgetflag((char*)"xn");
      /* Can we tab */
      Val(T_pt) = tgetflag((char*)"pt");
      Val(T_xt) = tgetflag((char*)"xt");
      /* do we have a meta? */
      Val(T_km) = tgetflag((char*)"km");
      Val(T_MT) = tgetflag((char*)"MT");
      /* Get the size */
      Val(T_co) = tgetnum((char*)"co");
      Val(T_li) = tgetnum((char*)"li");

      for (t = tstr; t->fName != NULL; t++) {
         term_alloc(el, t, tgetstr((char*)t->fName, &area));
      }
   }

   if (Val(T_co) < 2) {
      Val(T_co) = 80;           /* just in case */
   }

   if (Val(T_li) < 1) {
      Val(T_li) = 24;
   }

   el->fTerm.fSize.fV = Val(T_co);
   el->fTerm.fSize.fH = Val(T_li);

   term_setflags(el);

   /* get the correct window size */
   (void) term_get_size(el, &lins, &cols);

   if (term_change_size(el, lins, cols) == -1) {
      return -1;
   }
   (void) sigprocmask(SIG_SETMASK, &oset, NULL);
   term_bind_arrow(el);
   return i <= 0 ? -1 : 0;
} // term_set


/* term_get_size():
 *	Return the new window size in lines and cols, and
 *	true if the size was changed.
 */
el_protected int
term_get_size(EditLine_t* el, int* lins, int* cols) {
   *cols = Val(T_co);
   *lins = Val(T_li);

#ifdef TIOCGWINSZ
   {
      struct winsize ws;

      if (ioctl(el->fInFD, TIOCGWINSZ, (ioctl_t) &ws) != -1) {
         if (ws.ws_col) {
            *cols = ws.ws_col;
         }

         if (ws.ws_row) {
            *lins = ws.ws_row;
         }
      }
   }
#endif
#ifdef TIOCGSIZE
   {
      struct ttysize ts;

      if (ioctl(el->fInFD, TIOCGSIZE, (ioctl_t) &ts) != -1) {
         if (ts.ts_cols) {
            *cols = ts.ts_cols;
         }

         if (ts.ts_lines) {
            *lins = ts.ts_lines;
         }
      }
   }
#endif
   return Val(T_co) != *cols || Val(T_li) != *lins;
} // term_get_size


/* term_change_size():
 *	Change the size of the terminal
 */
el_protected int
term_change_size(EditLine_t* el, int lins, int cols) {
   /*
    * Just in case
    */
   Val(T_co) = (cols < 2) ? 80 : cols;
   Val(T_li) = (lins < 1) ? 24 : lins;

   /* re-make display buffers */
   if (term_rebuffer_display(el) == -1) {
      return -1;
   }
   re_clear_display(el);
   return 0;
}


/* term_init_arrow():
 *	Initialize the arrow key bindings from termcap
 */
el_private void
term_init_arrow(EditLine_t* el) {
   FKey_t* arrow = el->fTerm.fFKey;

   arrow[A_K_DN].fName = "down";
   arrow[A_K_DN].fKey = T_kd;
   arrow[A_K_DN].fFun.fCmd = ED_NEXT_HISTORY;
   arrow[A_K_DN].fType = XK_CMD;

   arrow[A_K_UP].fName = "up";
   arrow[A_K_UP].fKey = T_ku;
   arrow[A_K_UP].fFun.fCmd = ED_PREV_HISTORY;
   arrow[A_K_UP].fType = XK_CMD;

   arrow[A_K_LT].fName = "left";
   arrow[A_K_LT].fKey = T_kl;
   arrow[A_K_LT].fFun.fCmd = ED_PREV_CHAR;
   arrow[A_K_LT].fType = XK_CMD;

   arrow[A_K_RT].fName = "right";
   arrow[A_K_RT].fKey = T_kr;
   arrow[A_K_RT].fFun.fCmd = ED_NEXT_CHAR;
   arrow[A_K_RT].fType = XK_CMD;

   arrow[A_K_HO].fName = "home";
   arrow[A_K_HO].fKey = T_kh;
   arrow[A_K_HO].fFun.fCmd = ED_MOVE_TO_BEG;
   arrow[A_K_HO].fType = XK_CMD;

   arrow[A_K_EN].fName = "end";
   arrow[A_K_EN].fKey = T_at7;
   arrow[A_K_EN].fFun.fCmd = ED_MOVE_TO_END;
   arrow[A_K_EN].fType = XK_CMD;

   arrow[A_K_DE].fName = "del";
   arrow[A_K_DE].fKey = T_kD;
   arrow[A_K_DE].fFun.fCmd = ED_DELETE_NEXT_CHAR;      //EM_DELETE_OR_LIST;
   arrow[A_K_DE].fType = XK_CMD;
} // term_init_arrow


/* term_reset_arrow():
 *	Reset arrow key bindings
 */
el_private void
term_reset_arrow(EditLine_t* el) {
   FKey_t* arrow = el->fTerm.fFKey;
   static const char strA[] = { 033, '[', 'A', '\0' };
   static const char strB[] = { 033, '[', 'B', '\0' };
   static const char strC[] = { 033, '[', 'C', '\0' };
   static const char strD[] = { 033, '[', 'D', '\0' };
   static const char strH[] = { 033, '[', 'H', '\0' };
   static const char strF[] = { 033, '[', 'F', '\0' };
   static const char stOA[] = { 033, 'O', 'A', '\0' };
   static const char stOB[] = { 033, 'O', 'B', '\0' };
   static const char stOC[] = { 033, 'O', 'C', '\0' };
   static const char stOD[] = { 033, 'O', 'D', '\0' };
   static const char stOH[] = { 033, 'O', 'H', '\0' };
   static const char stOF[] = { 033, 'O', 'F', '\0' };

   key_add(el, strA, &arrow[A_K_UP].fFun, arrow[A_K_UP].fType);
   key_add(el, strB, &arrow[A_K_DN].fFun, arrow[A_K_DN].fType);
   key_add(el, strC, &arrow[A_K_RT].fFun, arrow[A_K_RT].fType);
   key_add(el, strD, &arrow[A_K_LT].fFun, arrow[A_K_LT].fType);
   key_add(el, strH, &arrow[A_K_HO].fFun, arrow[A_K_HO].fType);
   key_add(el, strF, &arrow[A_K_EN].fFun, arrow[A_K_EN].fType);
   key_add(el, stOA, &arrow[A_K_UP].fFun, arrow[A_K_UP].fType);
   key_add(el, stOB, &arrow[A_K_DN].fFun, arrow[A_K_DN].fType);
   key_add(el, stOC, &arrow[A_K_RT].fFun, arrow[A_K_RT].fType);
   key_add(el, stOD, &arrow[A_K_LT].fFun, arrow[A_K_LT].fType);
   key_add(el, stOH, &arrow[A_K_HO].fFun, arrow[A_K_HO].fType);
   key_add(el, stOF, &arrow[A_K_EN].fFun, arrow[A_K_EN].fType);
   key_add(el, stOF, &arrow[A_K_EN].fFun, arrow[A_K_EN].fType);

   if (el->fMap.fType == MAP_VI) {
      key_add(el, &strA[1], &arrow[A_K_UP].fFun, arrow[A_K_UP].fType);
      key_add(el, &strB[1], &arrow[A_K_DN].fFun, arrow[A_K_DN].fType);
      key_add(el, &strC[1], &arrow[A_K_RT].fFun, arrow[A_K_RT].fType);
      key_add(el, &strD[1], &arrow[A_K_LT].fFun, arrow[A_K_LT].fType);
      key_add(el, &strH[1], &arrow[A_K_HO].fFun, arrow[A_K_HO].fType);
      key_add(el, &strF[1], &arrow[A_K_EN].fFun, arrow[A_K_EN].fType);
      key_add(el, &stOA[1], &arrow[A_K_UP].fFun, arrow[A_K_UP].fType);
      key_add(el, &stOB[1], &arrow[A_K_DN].fFun, arrow[A_K_DN].fType);
      key_add(el, &stOC[1], &arrow[A_K_RT].fFun, arrow[A_K_RT].fType);
      key_add(el, &stOD[1], &arrow[A_K_LT].fFun, arrow[A_K_LT].fType);
      key_add(el, &stOH[1], &arrow[A_K_HO].fFun, arrow[A_K_HO].fType);
      key_add(el, &stOF[1], &arrow[A_K_EN].fFun, arrow[A_K_EN].fType);
   }
} // term_reset_arrow


/* term_set_arrow():
 *	Set an arrow key binding
 */
el_protected int
term_set_arrow(EditLine_t* el, char* name, KeyValue_t* fun, int type) {
   FKey_t* arrow = el->fTerm.fFKey;
   int i;

   for (i = 0; i < A_K_NKEYS; i++) {
      if (strcmp(name, arrow[i].fName) == 0) {
         arrow[i].fFun = *fun;
         arrow[i].fType = type;
         return 0;
      }
   }
   return -1;
}


/* term_clear_arrow():
 *	Clear an arrow key binding
 */
el_protected int
term_clear_arrow(EditLine_t* el, char* name) {
   FKey_t* arrow = el->fTerm.fFKey;
   int i;

   for (i = 0; i < A_K_NKEYS; i++) {
      if (strcmp(name, arrow[i].fName) == 0) {
         arrow[i].fType = XK_NOD;
         return 0;
      }
   }
   return -1;
}


/* term_print_arrow():
 *	Print the arrow key bindings
 */
el_protected void
term_print_arrow(EditLine_t* el, const char* name) {
   int i;
   FKey_t* arrow = el->fTerm.fFKey;

   for (i = 0; i < A_K_NKEYS; i++) {
      if (*name == '\0' || strcmp(name, arrow[i].fName) == 0) {
         if (arrow[i].fType != XK_NOD) {
            key_kprint(el, arrow[i].fName, &arrow[i].fFun,
                       arrow[i].fType);
         }
      }
   }
}


/* term_bind_arrow():
 *	Bind the arrow keys
 */
el_protected void
term_bind_arrow(EditLine_t* el) {
   ElAction_t* map;
   const ElAction_t* dmap;
   int i, j;
   char* p;
   FKey_t* arrow = el->fTerm.fFKey;

   /* Check if the components needed are initialized */
   if (el->fTerm.fBuf == NULL || el->fMap.fKey == NULL) {
      return;
   }

   map = el->fMap.fType == MAP_VI ? el->fMap.fAlt : el->fMap.fKey;
   dmap = el->fMap.fType == MAP_VI ? el->fMap.fVic : el->fMap.fEmacs;

   term_reset_arrow(el);

   for (i = 0; i < A_K_NKEYS; i++) {
      p = el->fTerm.fStr[arrow[i].fKey];

      if (p && *p) {
         j = (unsigned char) *p;

         /*
          * Assign the arrow keys only if:
          *
          * 1. They are multi-character arrow keys and the user
          *    has not re-assigned the leading character, or
          *    has re-assigned the leading character to be
          *	  ED_SEQUENCE_LEAD_IN
          * 2. They are single arrow keys pointing to an
          *    unassigned key.
          */
         if (arrow[i].fType == XK_NOD) {
            key_clear(el, map, p);
         } else {
            if (p[1] && (dmap[j] == map[j] ||
                         map[j] == ED_SEQUENCE_LEAD_IN)) {
               key_add(el, p, &arrow[i].fFun,
                       arrow[i].fType);
               map[j] = ED_SEQUENCE_LEAD_IN;
            } else if (map[j] == ED_UNASSIGNED) {
               key_clear(el, map, p);

               if (arrow[i].fType == XK_CMD) {
                  map[j] = arrow[i].fFun.fCmd;
               } else {
                  key_add(el, p, &arrow[i].fFun,
                          arrow[i].fType);
               }
            }
         }
      }
   }
} // term_bind_arrow


/* term_init_color():
 *	Initialize the color handling
 */
el_private void
term_init_color(EditLine_t* el) {
   int errcode;

   if ((el->fFlags & NO_TTY) || !isatty(1)) {
      // no TTY, no color.
      return;
   }

   if (ERR == setupterm(0, 1, &errcode)) {
      char* eldebug = getenv("EDITLINEDEBUG");
      if (eldebug != 0 && eldebug[0]) {
         fprintf(stderr, "ERROR initializing the terminal [TERM=%s]:\n", getenv("TERM"));
         switch (errcode) {
         case 1:
            fprintf(stderr,
                    "  Your terminal cannot be used for curses applications [code 1].\n"
                    "  Please reconfigure ROOT with --disable-editline, or get a better terminal.\n\n");
            break;
         case 0:
            fprintf(stderr,
                    "  the terminal could not be found, or it is a generic type [code 0].\n"
                    "  Please reconfigure ROOT with --disable-editline, or get a better terminal.\n\n");
            break;
         case -1:
            fprintf(stderr,
                    "  the terminfo database could not be found [code -1].\n"
                    "  Please make sure that it is accessible.\n\n");
            break;
         default:
            fprintf(stderr,
                    "  unknown curses error while setting up the terminal [code %d].\n\n",
                    errcode);
         }
      }
   }
}


/* term__gettermmanip():
 *	Retrieve the static terminal manipulation object
 */
el_private TTermManip&
term__gettermmanip() {
   static TTermManip tm; /* Terminal color manipulation */
   return tm;
}


/* term__resetcolor():
 *	Reset the color to its default value
 */
el_protected void
term__resetcolor() {
   TTermManip& tm = term__gettermmanip();
   tm.ResetTerm();
   term__flush();
}


/* term__putc():
 *	Add a character
 */
el_protected int
term__putc(int c) {
   return term__putcolorch(c, NULL);
}

/* term__atocolor():
 *      Get the color index for a color name.
 *      Name can be black, gray, blue,...
 *      or #rrbbgg or #rgb
 */
el_public int
term__atocolor(const char* name) {
   int attr = 0;
   std::string lowname(name);
   size_t lenname = strlen(name);
   for (size_t i = 0; i < lenname; ++i)
      lowname[i] = tolower(lowname[i]);

   if (lowname.find("bold") != std::string::npos
       || lowname.find("light") != std::string::npos)
      attr |= 0x2000;
   if (lowname.find("under") != std::string::npos)
      attr |= 0x4000;

   TTermManip& tm = term__gettermmanip();
   size_t poshash = lowname.find('#');
   size_t lenrgb = 0;
   if (poshash != std::string::npos) {
      int endrgb = poshash + 1;
      while ((lowname[endrgb] >= '0' && lowname[endrgb] <= '9')
              || (lowname[endrgb] >= 'a' && lowname[endrgb] <= 'f')) {
         ++endrgb;
      }
      lenrgb = endrgb - poshash - 1;
   }

   if (lenrgb == 3) {
      int rgb[3] = {0};
      for (int i = 0; i < 3; ++i) {
         rgb[i] = lowname[poshash + 1 + i] - '0';
         if (rgb[i] > 9) {
            rgb[i] = rgb[i] + '0' - 'a' + 10;
         }
         rgb[i] *= 16; // only upper 4 bits are set.
      }
      return attr | tm.GetColorIndex(rgb[0], rgb[1], rgb[2]);
   } else if (lenrgb == 6) {
      int rgb[3] = {0};
      for (int i = 0; i < 6; ++i) {
         int v = lowname[poshash + 1 + i] - '0';
         if (v > 9) {
            v = v + '0' - 'a' + 10;
         }
         if (i % 2 == 0) {
            v *= 16;
         }
         rgb[i / 2] += v;
      }
      return attr | tm.GetColorIndex(rgb[0], rgb[1], rgb[2]);
   } else {
      if (lowname.find("default") != std::string::npos) {
         return attr | 0xff;
      }

      static const char* colornames[] = {
         "black", "red", "green", "yellow",
         "blue", "magenta", "cyan", "white", 0
      };
      static const unsigned char colorrgb[][3] = {
         {0,0,0}, {127,0,0}, {0,127,0}, {127,127,0},
         {0,0,127}, {127,0,127}, {0,127,127}, {127,127,127},
         {0}
      };

      for (int i = 0; colornames[i]; ++i) {
         if (lowname.find(colornames[i]) != std::string::npos) {
            int boldify = 0;
            if (attr & 0x2000)
               boldify = 64;
            return attr | tm.GetColorIndex(colorrgb[i][0] + boldify,
                                           colorrgb[i][1] + boldify,
                                           colorrgb[i][2] + boldify);
         }
      }
   }
   fprintf(stderr, "editline / term__atocolor: cannot parse color %s!\n", name);
   return -1;
}


/* term__setcolor():
 *	Set terminal to a given foreground colour
 */
el_protected void
term__setcolor(int fgcol) {
   TTermManip& tm = term__gettermmanip();

   int idx = (fgcol & 0xff);
   if (idx == 0xff) {
      tm.SetDefaultColor();
   } else {
      tm.SetColor(idx);
   }

   if (fgcol != -1) {
      if (fgcol & 0x2000) {
         tm.StartBold();
      } else {
         tm.StopBold();
      }
      if (fgcol & 0x4000) {
         tm.StartUnderline();
      } else {
         tm.StopUnderline();
      }
   }

} // term__setcolor


/* term__putcolorch():
 *	Add a character with colour information
 */
el_protected int
term__putcolorch(int c, ElColor_t* color) {
   if (color != NULL && c != ' ') {
      term__setcolor(color->fForeColor);
   }

   int res = fputc(c, term_outfile);
   return res;

}


/* term__repaint():
 *	Repaint existing character at index (in order to display new colour attribute)
 */
el_protected void
term__repaint(EditLine_t* el, int index) {
   // store where cursor is currently (involves el)
   char* cursor = el->fLine.fCursor;

   int promptSize = el->fPrompt.fPos.fH;
   int oriCursor = el->fCursor.fH;
   int oriLine = el->fCursor.fV;

   // move to index of char to change
   el->fLine.fCursor = el->fLine.fBuffer + index;

   int line = (promptSize + index) / el->fTerm.fSize.fH;
   int hpos = (promptSize + index) % el->fTerm.fSize.fH;
   term_move_to_line(el, line);
   term_move_to_char(el, hpos);

   // rewrite char
   term_overwrite(el, el->fLine.fCursor,
                  el->fLine.fBufColor + index,
                  1);

   // move cursor back
   el->fLine.fCursor = cursor;
   term_move_to_line(el, oriLine);
   term_move_to_char(el, oriCursor);

   term__flush();
} // term__repaint


/* term__flush():
 *	Flush output
 */
el_protected void
term__flush(void) {
   (void) fflush(term_outfile);
}


/* term_telltc():
 *	Print the current termcap characteristics
 */
el_protected int
/*ARGSUSED*/
term_telltc(EditLine_t* el, int /*argc*/, const char** /*argv*/) {
   const struct TermCapStr_t* t;
   char** ts;
   char upbuf[EL_BUFSIZ];

   (void) fprintf(el->fOutFile, "\n\tYour terminal has the\n");
   (void) fprintf(el->fOutFile, "\tfollowing characteristics:\n\n");
   (void) fprintf(el->fOutFile, "\tIt has %d columns and %d lines\n",
                  Val(T_co), Val(T_li));
   (void) fprintf(el->fOutFile,
                  "\tIt has %s meta key\n", EL_HAS_META ? "a" : "no");
   (void) fprintf(el->fOutFile,
                  "\tIt can%suse tabs\n", EL_CAN_TAB ? " " : "not ");
   (void) fprintf(el->fOutFile, "\tIt %s automatic margins\n",
                  EL_HAS_AUTO_MARGINS ? "has" : "does not have");

   if (EL_HAS_AUTO_MARGINS) {
      (void) fprintf(el->fOutFile, "\tIt %s magic margins\n",
                     EL_HAS_MAGIC_MARGINS ? "has" : "does not have");
   }

   for (t = tstr, ts = el->fTerm.fStr; t->fName != NULL; t++, ts++) {
      (void) fprintf(el->fOutFile, "\t%25s (%s) == %s\n",
                     t->fLongName,
                     t->fName, *ts && **ts ?
                     key__decode_str(*ts, upbuf, "") : "(empty)");
   }
   (void) fputc('\n', el->fOutFile);
   return 0;
} // term_telltc


/* term_settc():
 *	Change the current terminal characteristics
 */
el_protected int
/*ARGSUSED*/
term_settc(EditLine_t* el, int /*argc*/, const char** argv) {
   const struct TermCapStr_t* ts;
   const struct TermCapVal_t* tv;
   const char* what, * how;

   if (argv == NULL || argv[1] == NULL || argv[2] == NULL) {
      return -1;
   }

   // gcc bug? it sees these access as non-const:
   what = argv[1];
   how = argv[2];

   /*
    * Do the strings first
    */
   for (ts = tstr; ts->fName != NULL; ts++) {
      if (strcmp(ts->fName, what) == 0) {
         break;
      }
   }

   if (ts->fName != NULL) {
      term_alloc(el, ts, how);
      term_setflags(el);
      return 0;
   }

   /*
    * Do the numeric ones second
    */
   for (tv = tval; tv->fName != NULL; tv++) {
      if (strcmp(tv->fName, what) == 0) {
         break;
      }
   }

   if (tv->fName != NULL) {
      if (tv == &tval[T_pt] || tv == &tval[T_km] ||
          tv == &tval[T_am] || tv == &tval[T_xn]) {
         if (strcmp(how, "yes") == 0) {
            el->fTerm.fVal[tv - tval] = 1;
         } else if (strcmp(how, "no") == 0) {
            el->fTerm.fVal[tv - tval] = 0;
         } else {
            (void) fprintf(el->fErrFile,
                           "settc: Bad value `%s'.\n", how);
            return -1;
         }
         term_setflags(el);

         if (term_change_size(el, Val(T_li), Val(T_co)) == -1) {
            return -1;
         }
         return 0;
      } else {
         long i;
         char* ep;

         i = strtol(how, &ep, 10);

         if (*ep != '\0') {
            (void) fprintf(el->fErrFile,
                           "settc: Bad value `%s'.\n", how);
            return -1;
         }
         el->fTerm.fVal[tv - tval] = (int) i;
         el->fTerm.fSize.fV = Val(T_co);
         el->fTerm.fSize.fH = Val(T_li);

         if (tv == &tval[T_co] || tv == &tval[T_li]) {
            if (term_change_size(el, Val(T_li), Val(T_co))
                == -1) {
               return -1;
            }
         }
         return 0;
      }
   }
   return -1;
} // term_settc


/* term_echotc():
 *	Print the termcap string out with variable substitution
 */
el_protected int
/*ARGSUSED*/
term_echotc(EditLine_t* el, int /*argc*/, const char** argv) {
   char* cap, * scap, * ep;
   int arg_need, arg_cols, arg_rows;
   int verbose = 0, silent = 0;
   char* area;
   static const char fmts[] = "%s\n", fmtd[] = "%d\n";
   const struct TermCapStr_t* t;
   char buf[TC_BUFSIZE];
   long i;

   area = buf;

   if (argv == NULL || argv[1] == NULL) {
      return -1;
   }
   argv++;

   if (argv[0][0] == '-') {
      switch (argv[0][1]) {
      case 'v':
         verbose = 1;
         break;
      case 's':
         silent = 1;
         break;
      default:
         /* stderror(ERR_NAME | ERR_TCUSAGE); */
         break;
      }
      argv++;
   }

   if (!*argv || *argv[0] == '\0') {
      return 0;
   }

   if (strcmp(*argv, "tabs") == 0) {
      (void) fprintf(el->fOutFile, fmts, EL_CAN_TAB ? "yes" : "no");
      return 0;
   } else if (strcmp(*argv, "meta") == 0) {
      (void) fprintf(el->fOutFile, fmts, Val(T_km) ? "yes" : "no");
      return 0;
   } else if (strcmp(*argv, "xn") == 0) {
      (void) fprintf(el->fOutFile, fmts, EL_HAS_MAGIC_MARGINS ?
                     "yes" : "no");
      return 0;
   } else if (strcmp(*argv, "am") == 0) {
      (void) fprintf(el->fOutFile, fmts, EL_HAS_AUTO_MARGINS ?
                     "yes" : "no");
      return 0;
   } else if (strcmp(*argv, "baud") == 0) {
#ifdef notdef
      int i;

      for (i = 0; baud_rate[i].b_name != NULL; i++) {
         if (el->fTTY.t_speed == baud_rate[i].b_rate) {
            (void) fprintf(el->fOutFile, fmts,
                           baud_rate[i].b_name);
            return 0;
         }
      }
      (void) fprintf(el->fOutFile, fmtd, 0);
#else
      (void) fprintf(el->fOutFile, fmtd, (int)el->fTTY.t_speed);
#endif
      return 0;
   } else if (strcmp(*argv, "rows") == 0 || strcmp(*argv, "lines") == 0) {
      (void) fprintf(el->fOutFile, fmtd, Val(T_li));
      return 0;
   } else if (strcmp(*argv, "cols") == 0) {
      (void) fprintf(el->fOutFile, fmtd, Val(T_co));
      return 0;
   }

   /*
    * Try to use our local definition first
    */
   scap = NULL;

   for (t = tstr; t->fName != NULL; t++) {
      if (strcmp(t->fName, *argv) == 0) {
         scap = el->fTerm.fStr[t - tstr];
         break;
      }
   }

   if (t->fName == NULL) {
      scap = tgetstr((char*)*argv, &area);
   }

   if (!scap || scap[0] == '\0') {
      if (!silent) {
         (void) fprintf(el->fErrFile,
                        "echotc: Termcap parameter `%s' not found.\n",
                        *argv);
      }
      return -1;
   }

   /*
    * Count home many values we need for this capability.
    */
   for (cap = scap, arg_need = 0; *cap; cap++) {
      if (*cap == '%') {
         switch (*++cap) {
         case 'd':
         case '2':
         case '3':
         case '.':
         case '+':
            arg_need++;
            break;
         case '%':
         case '>':
         case 'i':
         case 'r':
         case 'n':
         case 'B':
         case 'D':
            break;
         default:

            /*
             * hpux has lot's of them...
             */
            if (verbose) {
               (void) fprintf(el->fErrFile,
                              "echotc: Warning: unknown termcap %% `%c'.\n",
                              *cap);
            }
            /* This is bad, but I won't complain */
            break;
         } // switch
      }
   }

   switch (arg_need) {
   case 0:
      argv++;

      if (*argv && *argv[0]) {
         if (!silent) {
            (void) fprintf(el->fErrFile,
                           "echotc: Warning: Extra argument `%s'.\n",
                           *argv);
         }
         return -1;
      }
      (void) tputs(scap, 1, term__putc);
      break;
   case 1:
      argv++;

      if (!*argv || *argv[0] == '\0') {
         if (!silent) {
            (void) fprintf(el->fErrFile,
                           "echotc: Warning: Missing argument.\n");
         }
         return -1;
      }
      arg_cols = 0;
      i = strtol(*argv, &ep, 10);

      if (*ep != '\0' || i < 0) {
         if (!silent) {
            (void) fprintf(el->fErrFile,
                           "echotc: Bad value `%s' for rows.\n",
                           *argv);
         }
         return -1;
      }
      arg_rows = (int) i;
      argv++;

      if (*argv && *argv[0]) {
         if (!silent) {
            (void) fprintf(el->fErrFile,
                           "echotc: Warning: Extra argument `%s'.\n",
                           *argv);
         }
         return -1;
      }
      (void) tputs(tgoto(scap, arg_cols, arg_rows), 1, term__putc);
      break;
   default:

      /* This is wrong, but I will ignore it... */
      if (verbose) {
         (void) fprintf(el->fErrFile,
                        "echotc: Warning: Too many required arguments (%d).\n",
                        arg_need);
      }
   /* FALLTHROUGH */
   case 2:
      argv++;

      if (!*argv || *argv[0] == '\0') {
         if (!silent) {
            (void) fprintf(el->fErrFile,
                           "echotc: Warning: Missing argument.\n");
         }
         return -1;
      }
      i = strtol(*argv, &ep, 10);

      if (*ep != '\0' || i < 0) {
         if (!silent) {
            (void) fprintf(el->fErrFile,
                           "echotc: Bad value `%s' for cols.\n",
                           *argv);
         }
         return -1;
      }
      arg_cols = (int) i;
      argv++;

      if (!*argv || *argv[0] == '\0') {
         if (!silent) {
            (void) fprintf(el->fErrFile,
                           "echotc: Warning: Missing argument.\n");
         }
         return -1;
      }
      i = strtol(*argv, &ep, 10);

      if (*ep != '\0' || i < 0) {
         if (!silent) {
            (void) fprintf(el->fErrFile,
                           "echotc: Bad value `%s' for rows.\n",
                           *argv);
         }
         return -1;
      }
      arg_rows = (int) i;

      if (*ep != '\0') {
         if (!silent) {
            (void) fprintf(el->fErrFile,
                           "echotc: Bad value `%s'.\n", *argv);
         }
         return -1;
      }
      argv++;

      if (*argv && *argv[0]) {
         if (!silent) {
            (void) fprintf(el->fErrFile,
                           "echotc: Warning: Extra argument `%s'.\n",
                           *argv);
         }
         return -1;
      }
      (void) tputs(tgoto(scap, arg_cols, arg_rows), arg_rows, term__putc);
      break;
   } // switch
   return 0;
} // term_echotc
