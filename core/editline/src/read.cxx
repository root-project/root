// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: read.c,v 1.19 2001/01/10 07:45:41 jdolecek Exp $	*/

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
 * read.c: Clean this junk up! This is horrible code.
 *	   Terminal read functions
 */
#include "sys.h"
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include "el.h"
#include "enhance.h"


#define OKCMD -1

el_private int read__fixio(int, int);
el_private int read_preread(EditLine*);
el_private int read_getcmd(EditLine*, el_action_t*, char*);
el_private int read_char(EditLine*, char*);

#ifdef DEBUG_EDIT
el_private void
read_debug(EditLine* el) {
   if (el->el_line.cursor > el->el_line.lastchar) {
      (void) fprintf(el->el_errfile, "cursor > lastchar\r\n");
   }

   if (el->el_line.cursor < el->el_line.buffer) {
      (void) fprintf(el->el_errfile, "cursor < buffer\r\n");
   }

   if (el->el_line.cursor > el->el_line.limit) {
      (void) fprintf(el->el_errfile, "cursor > limit\r\n");
   }

   if (el->el_line.lastchar > el->el_line.limit) {
      (void) fprintf(el->el_errfile, "lastchar > limit\r\n");
   }

   if (el->el_line.limit != &el->el_line.buffer[EL_BUFSIZ - 2]) {
      (void) fprintf(el->el_errfile, "limit != &buffer[EL_BUFSIZ-2]\r\n");
   }
} // read_debug


#endif /* DEBUG_EDIT */


/* read__fixio():
 *	Try to recover from a read error
 */
/* ARGSUSED */
el_private int
read__fixio(int /*fd*/, int e) {
   switch (e) {
   case - 1:                    /* Make sure that the code is reachable */

#ifdef EWOULDBLOCK
   case EWOULDBLOCK:
# ifndef TRY_AGAIN
#  define TRY_AGAIN
# endif
#endif /* EWOULDBLOCK */

#if defined(POSIX) && defined(EAGAIN)
# if defined(EWOULDBLOCK) && EWOULDBLOCK != EAGAIN
   case EAGAIN:
#  ifndef TRY_AGAIN
#   define TRY_AGAIN
#  endif
# endif /* EWOULDBLOCK && EWOULDBLOCK != EAGAIN */
#endif /* POSIX && EAGAIN */

      e = 0;
#ifdef TRY_AGAIN
# if defined(F_SETFL) && defined(O_NDELAY)

      if ((e = fcntl(fd, F_GETFL, 0)) == -1) {
         return -1;
      }

      if (fcntl(fd, F_SETFL, e & ~O_NDELAY) == -1) {
         return -1;
      } else {
         e = 1;
      }
# endif /* F_SETFL && O_NDELAY */

# ifdef FIONBIO
      {
         int zero = 0;

         if (ioctl(fd, FIONBIO, (ioctl_t) &zero) == -1) {
            return -1;
         } else {
            e = 1;
         }
      }
# endif /* FIONBIO */

#endif /* TRY_AGAIN */
      return e ? 0 : -1;

   case EINTR:
      return 0;

   default:
      return -1;
   } // switch
} // read__fixio


/* read_preread():
 *	Try to read the stuff in the input queue;
 */
el_private int
read_preread(EditLine* el) {
   int chrs = 0;

   if (el->el_chared.c_macro.nline) {
      el_free((ptr_t) el->el_chared.c_macro.nline);
      el->el_chared.c_macro.nline = NULL;
   }

   if (el->el_tty.t_mode == ED_IO) {
      return 0;
   }

#ifdef FIONREAD
      (void) ioctl(el->el_infd, FIONREAD, (ioctl_t) &chrs);

   if (chrs > 0) {
      char buf[EL_BUFSIZ];

      chrs = read(el->el_infd, buf,
                  (size_t) MIN(chrs, EL_BUFSIZ - 1));

      if (chrs > 0) {
         buf[chrs] = '\0';
         el->el_chared.c_macro.nline = strdup(buf);
         el_push(el, el->el_chared.c_macro.nline);
      }
   }
#endif /* FIONREAD */

   return chrs > 0;
} // read_preread


/* el_push():
 *	Push a macro
 */
el_public void
el_push(EditLine* el, const char* str) {
   c_macro_t* ma = &el->el_chared.c_macro;

   if (str != NULL && ma->level + 1 < EL_MAXMACRO) {
      ma->level++;
      /* LINTED const cast */
      ma->macro[ma->level] = (char*) str;
   } else {
      term_beep(el);
      term__flush();
   }
}


/* read_getcmd():
 *	Return next command from the input stream.
 */
el_private int
read_getcmd(EditLine* el, el_action_t* cmdnum, char* ch) {
   el_action_t cmd = ED_UNASSIGNED;
   int num;

   while (cmd == ED_UNASSIGNED || cmd == ED_SEQUENCE_LEAD_IN) {
      if ((num = el_getc(el, ch)) != 1) {               /* if EOF or error */
         return num;
      }

#ifdef  KANJI

      if ((*ch & 0200)) {
         el->el_state.metanext = 0;
         cmd = CcViMap[' '];
         break;
      } else
#endif /* KANJI */

      if (el->el_state.metanext) {
         el->el_state.metanext = 0;
         *ch |= 0200;
      }
      cmd = el->el_map.current[(unsigned char) *ch];

      if (cmd == ED_SEQUENCE_LEAD_IN) {
         key_value_t val;

         switch (key_get(el, ch, &val)) {
         case XK_CMD:
            cmd = val.cmd;
            break;
         case XK_STR:
            el_push(el, val.str);
            break;
#ifdef notyet
         case XK_EXE:
            /* XXX: In the future to run a user function */
            RunCommand(val.str);
            break;
#endif
         default:
            EL_ABORT((el->el_errfile, "Bad XK_ type \n"));
            break;
         } // switch
      }

      if (el->el_map.alt == NULL) {
         el->el_map.current = el->el_map.key;
      }
   }
   *cmdnum = cmd;
   return OKCMD;
} // read_getcmd


/* read_char():
 *	Read a character from the tty.
 *	Initialise the character colour to NULL.
 */
el_private int
read_char(EditLine* el, char* cp) {
   int num_read;
   int tried = 0;

   while ((num_read = read(el->el_infd, cp, 1)) == -1) {
      if (!tried && read__fixio(el->el_infd, errno) == 0) {
         tried = 1;
      } else {
         *cp = '\0';
         return -1;
      }
   }
   // don't do this - "new" char may be a command char e.g <- or ->
   // set the colour of the newly read in char to null
   //el->el_line.bufcolor[el->el_line.cursor - el->el_line.buffer] = -1;

   return num_read;
} // read_char


/* el_getc():
 *	Read a character
 */
el_public int
el_getc(EditLine* el, char* cp) {
   int num_read;
   c_macro_t* ma = &el->el_chared.c_macro;

   term__flush();

   for ( ; ;) {
      if (ma->level < 0) {
         if (!read_preread(el)) {
            break;
         }
      }

      if (ma->level < 0) {
         break;
      }

      if (*ma->macro[ma->level] == 0) {
         ma->level--;
         continue;
      }
      *cp = *ma->macro[ma->level]++ & 0377;

      if (*ma->macro[ma->level] == 0) {                 /* Needed for QuoteMode
                                                         * On */
         ma->level--;
      }
      return 1;
   }

#ifdef DEBUG_READ
      (void) fprintf(el->el_errfile, "Turning raw mode on\n");
#endif /* DEBUG_READ */

   if (tty_rawmode(el) < 0) {   /* make sure the tty is set up correctly */
      return 0;
   }

#ifdef DEBUG_READ
      (void) fprintf(el->el_errfile, "Reading a character\n");
#endif /* DEBUG_READ */
   num_read = read_char(el, cp);
#ifdef DEBUG_READ
      (void) fprintf(el->el_errfile, "Got it %c\n", *cp);
#endif /* DEBUG_READ */
   return num_read;
} // el_getc


int
el_chop_at_newline(EditLine* el) {
   //fprintf( el->el_errfile, "el_chop_at_newline() 1:[%s]\n", el->el_line.buffer );
   char* str = el->el_line.buffer;

   if (str) {
      for ( ; str <= el->el_line.lastchar; ++str) {
         if (*str != '\n' && *str != '\r') {
            continue;
         }
         //el->el_line.lastchar = str; // ??? yes??? no???
         *str = '\0';
      }
   }
   //fprintf( el->el_errfile, "el_chop_at_newline() 2:[%s]\n", el->el_line.buffer );
   return strlen(el->el_line.buffer);
}


el_public const char*
el_gets(EditLine* el, int* nread) {
   int retval;
   el_action_t cmdnum = 0;
   int num;                     /* how many chars we have read at NL */
   char ch;
#ifdef FIONREAD
   c_macro_t* ma = &el->el_chared.c_macro;
#endif /* FIONREAD */

   //fprintf( el->el_errfile, "el_gets()\n" );

   if (el->el_flags & NO_TTY) {
      char* cp = el->el_line.buffer;
      size_t idx;
      size_t numRead = 0;

      while ((numRead = read_char(el, cp)) == 1) {
         /* make sure there is space for next character */
         if (cp + 1 >= el->el_line.limit) {
            idx = (cp - el->el_line.buffer);

            if (!ch_enlargebufs(el, 2)) {
               break;
            }
            cp = &el->el_line.buffer[idx];
         }
         cp++;

         if (cp[-1] == '\r' || cp[-1] == '\n') {
            // cp[-1] = '\0';
            // ^^^ added by stephan: this returning
            // the newline is tedious on clients
            // and in contrary to common STL
            // usage.
            break;
         }
      }

      if (!numRead) {
         // singal EOF by count > 0 but line==""
         *cp = 0;
         cp++;
         strcpy(cp, "EOF");
         cp += 3;
      }

      el->el_line.cursor = el->el_line.lastchar = cp;
      *cp = '\0';

      if (nread) {
         *nread = el->el_line.cursor - el->el_line.buffer;
      }
      return el->el_line.buffer;
   }

   if (el->el_flags & EDIT_DISABLED) {
      // fprintf(stderr, "el_gets() EDIT_DISABLED block\n" );
      char* cp = el->el_line.buffer;
      size_t idx;

      term__flush();

      while (read_char(el, cp) == 1) {
         /* make sure there is space next character */
         if (cp + 1 >= el->el_line.limit) {
            idx = (cp - el->el_line.buffer);

            if (!ch_enlargebufs(el, 2)) {
               break;
            }
            cp = &el->el_line.buffer[idx];
         }
         cp++;

         if (cp[-1] == '\r' || cp[-1] == '\n') {
            // cp[-1] = '\0';
            // ^^^ added by stephan: this returning
            // the newline is tedious on clients
            // and in contrary to common STL
            // usage.
            break;
         }
      }
      *cp = '\0';
      el->el_line.cursor = el->el_line.lastchar = cp;

      if (nread) {
         *nread = el->el_line.cursor - el->el_line.buffer;
      }
      return el->el_line.buffer;
   }

   //for (num = OKCMD; num == OKCMD;) {	/* while still editing this line */
   //num = OKCMD;

#ifdef DEBUG_EDIT
   read_debug(el);
#endif /* DEBUG_EDIT */

   /* if EOF or error */
   if ((num = read_getcmd(el, &cmdnum, &ch)) != OKCMD) {
#ifdef DEBUG_READ
         (void) fprintf(el->el_errfile,
                        "Returning from el_gets %d\n", num);
#endif /* DEBUG_READ */
      //break;
   }

   if ((int) cmdnum >= el->el_map.nfunc) {              /* BUG CHECK command */
#ifdef DEBUG_EDIT
         (void) fprintf(el->el_errfile,
                        "ERROR: illegal command from key 0%o\r\n", ch);
#endif /* DEBUG_EDIT */
      //continue;	/* try again */
   }
   /* now do the real command */
#ifdef DEBUG_READ
   {
      el_bindings_t* b;

      for (b = el->el_map.help; b->name; b++) {
         if (b->func == cmdnum) {
            break;
         }
      }

      if (b->name) {
         (void) fprintf(el->el_errfile,
                        "Executing %s\n", b->name);
      } else {
         (void) fprintf(el->el_errfile,
                        "Error command = %d\n", cmdnum);
      }
   }
#endif /* DEBUG_READ */
   retval = (*el->el_map.func[cmdnum])(el, ch);

   /* save the last command here */
   el->el_state.lastcmd = cmdnum;

   /* use any return value */
   switch (retval) {
   case CC_CURSOR:
      el->el_state.argument = 1;
      el->el_state.doingarg = 0;
      re_refresh_cursor(el);
      break;

   case CC_REDISPLAY:
      re_clear_lines(el);
      re_clear_display(el);
   /* FALLTHROUGH */

   case CC_REFRESH:
      el->el_state.argument = 1;
      el->el_state.doingarg = 0;
      re_refresh(el);
      break;

   case CC_REFRESH_BEEP:
      el->el_state.argument = 1;
      el->el_state.doingarg = 0;
      re_refresh(el);
      term_beep(el);
      break;

   case CC_NORM:                /* normal char */
      el->el_state.argument = 1;
      el->el_state.doingarg = 0;
      num = el->el_line.lastchar - el->el_line.buffer;                                  // LOUISE ret val
      break;

   case CC_ARGHACK:                     /* Suggested by Rich Salz */
      /* <rsalz@pineapple.bbn.com> */
      break;                    /* keep going... */

   case CC_EOF:                 /* end of file typed */
      num = 0;
      break;

   case CC_NEWLINE:                     /* normal end of line */
      num = el->el_line.lastchar - el->el_line.buffer;

      if (0 == num) {
         // Added by stephan, so that
         // the readline compat layer
         // can know the difference
         // between an empty line
         // and EOF.
         el->el_line.lastchar = el->el_line.buffer;
         *el->el_line.buffer = '\0';
         num = 1;
      }
      //printf( "el_gets() CC_NEWLINE! num=%d\n", num );
      break;

   case CC_FATAL:               /* fatal error, reset to known state */
#ifdef DEBUG_READ
         (void) fprintf(el->el_errfile,
                        "*** editor fatal ERROR ***\r\n\n");
#endif /* DEBUG_READ */
      /* put (real) cursor in a known place */
      re_clear_display(el);                     /* reset the display stuff */
      ch_reset(el);                     /* reset the input pointers */
      re_refresh(el);                   /* print the prompt again */
      el->el_state.argument = 1;
      el->el_state.doingarg = 0;
      break;

   case CC_ERROR:
   default:                     /* functions we don't know about */
#ifdef DEBUG_READ
         (void) fprintf(el->el_errfile,
                        "*** editor ERROR ***\r\n\n");
#endif /* DEBUG_READ */
      el->el_state.argument = 1;
      el->el_state.doingarg = 0;
      term_beep(el);
      term__flush();
      break;

   } // switch
     // i think its this one
     //}

/*         printf( "el_gets() num=%d\n", num ); */
   /* make sure the tty is set up correctly */
   //(void) tty_cookedmode(el);
   term__flush();               /* flush any buffered output */

   //if (el->el_flags & HANDLE_SIGNALS)
   //	sig_clr(el);
   if (nread) {
      *nread = num;
   }

   if (retval == 1) {    /* enter key pressed - add a newline */
      *el->el_line.lastchar = '\n';
      el->el_line.lastchar++;
      *el->el_line.lastchar = 0;

      return num ? el->el_line.buffer : NULL;
   }

   c_macro_t* ma = &el->el_chared.c_macro;

   if (retval == CC_REFRESH && ma->level >= 0 && ma->macro[ma->level]) {
      int subnread;
      el_gets(el, &subnread);

      if (nread) {
         *nread += subnread;
      }
   } else {
      // this happens for every char - need to add some logic to enhance to make it not check if not a word etc
      // [a-zA-Z]+[0-9].
      highlightKeywords(el);

      // if the cursor is at some point in the middle of the buffer, check for brackets
      if (el->el_line.cursor <= el->el_line.lastchar) {
         matchParentheses(el);
      }


      /* '\a' indicates entry is not complete */
      *el->el_line.lastchar = '\a';
   }

   return num ? el->el_line.buffer : NULL;
} // el_gets


el_public const char*
el_gets_newline(EditLine* el, int* nread) {
   if (el->el_flags & HANDLE_SIGNALS) {
      sig_set(el);
   }
   re_clear_display(el);        /* reset the display stuff */

   /* '\a' is used to signal that we re-entered this function without newline being hit. */
   if (*el->el_line.lastchar == '\a') {
      // Remove the '\a'
      // by letting getc overwrite it!
   } else {
      // Only reset the buffer if we edit a whole new line
      ch_reset(el);
   }

   if (!(el->el_flags & NO_TTY)) {
      re_refresh(el);                   /* print the prompt */
   }
   term__flush();

   if (nread) {
      *nread = 0;
   }
   return NULL;
} // el_gets_newline


el_public bool
el_eof(EditLine* el) {
   return !el->el_line.buffer[0] && !strcmp(el->el_line.buffer + 1, "EOF");
}


/**
   el_public const char *
   el_gets(EditLine *el, int *nread)
   {
        el_gets( el, nread );
 *nread = el_chop_at_newline(el);
        return el->el_line.buffer;
   }
 */
