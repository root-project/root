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
el_private int read_preread(EditLine_t*);
el_private int read_getcmd(EditLine_t*, ElAction_t*, char*);
el_private int read_char(EditLine_t*, char*);

#ifdef DEBUG_EDIT
el_private void
read_debug(EditLine_t* el) {
   if (el->fLine.fCursor > el->fLine.fLastChar) {
      (void) fprintf(el->fErrFile, "cursor > lastchar\r\n");
   }

   if (el->fLine.fCursor < el->fLine.fBuffer) {
      (void) fprintf(el->fErrFile, "cursor < buffer\r\n");
   }

   if (el->fLine.fCursor > el->fLine.fLimit) {
      (void) fprintf(el->fErrFile, "cursor > limit\r\n");
   }

   if (el->fLine.fLastChar > el->fLine.fLimit) {
      (void) fprintf(el->fErrFile, "lastchar > limit\r\n");
   }

   if (el->fLine.fLimit != &el->fLine.fBuffer[EL_BUFSIZ - 2]) {
      (void) fprintf(el->fErrFile, "limit != &buffer[EL_BUFSIZ-2]\r\n");
   }
} // read_debug


#endif /* DEBUG_EDIT */


/* read__fixio():
 *	Try to recover from a read error
 */
/* ARGSUSED */
el_private int
read__fixio(int 
# if (defined(F_SETFL) && defined(O_NDELAY)) \
     || defined(FIONBIO)
            fd
#endif
            , int e) {
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
read_preread(EditLine_t* el) {
   int chrs = 0;

   if (el->fCharEd.fMacro.fNLine) {
      el_free((ptr_t) el->fCharEd.fMacro.fNLine);
      el->fCharEd.fMacro.fNLine = NULL;
   }

   if (el->fTTY.t_mode == ED_IO) {
      return 0;
   }

#ifdef FIONREAD
   if (!el->fIn) {
      (void) ioctl(el->fInFD, FIONREAD, (ioctl_t) &chrs);

      if (chrs > 0) {
         char buf[EL_BUFSIZ];

         chrs = read(el->fInFD, buf,
                     (size_t) MIN(chrs, EL_BUFSIZ - 1));

         if (chrs > 0) {
            buf[chrs] = '\0';
            el->fCharEd.fMacro.fNLine = strdup(buf);
            el_push(el, el->fCharEd.fMacro.fNLine);
         }
      }
   }
#endif /* FIONREAD */

   return chrs > 0;
} // read_preread


/* el_push():
 *	Push a macro
 */
el_public void
el_push(EditLine_t* el, const char* str) {
   CMacro_t* ma = &el->fCharEd.fMacro;

   if (str != NULL && ma->fLevel + 1 < EL_MAXMACRO) {
      ma->fLevel++;
      /* LINTED const cast */
      ma->fMacro[ma->fLevel] = (char*) str;
   } else {
      term_beep(el);
      term__flush();
   }
}


/* read_getcmd():
 *	Return next command from the input stream.
 */
el_private int
read_getcmd(EditLine_t* el, ElAction_t* cmdnum, char* ch) {
   ElAction_t cmd = ED_UNASSIGNED;
   int num;

   while (cmd == ED_UNASSIGNED || cmd == ED_SEQUENCE_LEAD_IN) {
      if ((num = el_getc(el, ch)) != 1) {               /* if EOF or error */
         return num;
      }

#ifdef  KANJI

      if ((*ch & 0200)) {
         el->fState.fMetaNext = 0;
         cmd = CcViMap[' '];
         break;
      } else
#endif /* KANJI */

      if (el->fState.fMetaNext) {
         el->fState.fMetaNext = 0;
         *ch |= 0200;
      }
      // Coverity is complaining that the value of ch comes from the user
      // and nowhere do we check its value. But that's fine: it's 0<=ch<255,
      // and fCurrent has 256 entries.
      // coverity[data_index]
      // coverity[tainted_data]
      cmd = el->fMap.fCurrent[(unsigned char) *ch];

      if (cmd == ED_SEQUENCE_LEAD_IN) {
         KeyValue_t val;

         switch (key_get(el, ch, &val)) {
         case XK_CMD:
            cmd = val.fCmd;
            break;
         case XK_STR:
            el_push(el, val.fStr);
            break;
#ifdef notyet
         case XK_EXE:
            /* XXX: In the future to run a user function */
            RunCommand(val.fStr);
            break;
#endif
         default:
            EL_ABORT((el->fErrFile, "Bad XK_ type \n"));
            break;
         } // switch
      }

      if (el->fMap.fAlt == NULL) {
         el->fMap.fCurrent = el->fMap.fKey;
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
read_char(EditLine_t* el, char* cp) {
   int num_read;
   int tried = 0;

   if (!el->fIn) {
      while ((num_read = read(el->fInFD, cp, 1)) == -1) {
         if (!tried && read__fixio(el->fInFD, errno) == 0) {
            tried = 1;
         } else {
            *cp = '\0';
            return -1;
         }
      }
   } else {
      if (feof(el->fIn)) {
         *cp = 0;
         return 0;
      }
      int chread = fgetc(el->fIn);
#ifdef DEBUG_READ
      if (chread < 0 || chread > 255) {
         fprintf(el->fErrFile,
                 "Read unexpected character value %d\n", chread);
      }
#endif
      *cp = (char)chread;
      num_read = 1;
   }
   // don't do this - "new" char may be a command char e.g <- or ->
   // set the colour of the newly read in char to null
   //el->fLine.fBufColor[el->fLine.fCursor - el->fLine.fBuffer] = -1;

   return num_read;
} // read_char


/* el_getc():
 *	Read a character
 */
el_public int
el_getc(EditLine_t* el, char* cp) {
   int num_read;
   CMacro_t* ma = &el->fCharEd.fMacro;

   term__flush();

   for ( ; ;) {
      if (ma->fLevel < 0) {
         if (!read_preread(el)) {
            break;
         }
      }

      if (ma->fLevel < 0) {
         break;
      }

      if (*ma->fMacro[ma->fLevel] == 0) {
         ma->fLevel--;
         continue;
      }
      *cp = *ma->fMacro[ma->fLevel]++ & 0377;

      if (*ma->fMacro[ma->fLevel] == 0) {                 /* Needed for QuoteMode
                                                         * On */
         ma->fLevel--;
      }
      return 1;
   }

#ifdef DEBUG_READ
      (void) fprintf(el->fErrFile, "Turning raw mode on\n");
#endif /* DEBUG_READ */

   if (tty_rawmode(el) < 0) {   /* make sure the tty is set up correctly */
      return 0;
   }

#ifdef DEBUG_READ
      (void) fprintf(el->fErrFile, "Reading a character\n");
#endif /* DEBUG_READ */
   num_read = read_char(el, cp);
#ifdef DEBUG_READ
      (void) fprintf(el->fErrFile, "Got it %c\n", *cp);
#endif /* DEBUG_READ */
   return num_read;
} // el_getc


int
el_chop_at_newline(EditLine_t* el) {
   //fprintf( el->fErrFile, "el_chop_at_newline() 1:[%s]\n", el->fLine.fBuffer );
   char* str = el->fLine.fBuffer;

   if (str) {
      for ( ; str <= el->fLine.fLastChar; ++str) {
         if (*str != '\n' && *str != '\r') {
            continue;
         }
         //el->fLine.fLastChar = str; // ??? yes??? no???
         *str = '\0';
      }
   }
   //fprintf( el->fErrFile, "el_chop_at_newline() 2:[%s]\n", el->fLine.fBuffer );
   return strlen(el->fLine.fBuffer);
}


el_public const char*
el_gets(EditLine_t* el, int* nread) {
   int retval;
   ElAction_t cmdnum = 0;
   int num;                     /* how many chars we have read at NL */
   char ch;
#ifdef FIONREAD
   CMacro_t* ma = &el->fCharEd.fMacro;
#endif /* FIONREAD */

   //fprintf( el->fErrFile, "el_gets()\n" );

   if (el->fFlags & NO_TTY) {
      char* cp = el->fLine.fBuffer;
      size_t idx;
      size_t numRead = 0;

      while ((numRead = read_char(el, cp)) == 1) {
         /* make sure there is space for next character */
         if (cp + 4 >= el->fLine.fLimit) { // "+4" for "EOF" below
            idx = (cp - el->fLine.fBuffer);

            if (!ch_enlargebufs(el, 2)) {
               break;
            }
            cp = &el->fLine.fBuffer[idx];
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

      el->fLine.fCursor = el->fLine.fLastChar = cp;
      *cp = '\0';

      if (nread) {
         *nread = el->fLine.fCursor - el->fLine.fBuffer;
      }
      return el->fLine.fBuffer;
   }

   if (el->fFlags & EDIT_DISABLED) {
      // fprintf(stderr, "el_gets() EDIT_DISABLED block\n" );
      char* cp = el->fLine.fBuffer;
      size_t idx;

      term__flush();

      while (read_char(el, cp) == 1) {
         /* make sure there is space next character */
         if (cp + 1 >= el->fLine.fLimit) {
            idx = (cp - el->fLine.fBuffer);

            if (!ch_enlargebufs(el, 2)) {
               break;
            }
            cp = &el->fLine.fBuffer[idx];
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
      el->fLine.fCursor = el->fLine.fLastChar = cp;

      if (nread) {
         *nread = el->fLine.fCursor - el->fLine.fBuffer;
      }
      return el->fLine.fBuffer;
   }

   //for (num = OKCMD; num == OKCMD;) {	/* while still editing this line */
   //num = OKCMD;

#ifdef DEBUG_EDIT
   read_debug(el);
#endif /* DEBUG_EDIT */

   /* if EOF or error */
   // See coverity[data_index] in read_getcmd():
   // coverity[tainted_data]
   if ((num = read_getcmd(el, &cmdnum, &ch)) != OKCMD) {
#ifdef DEBUG_READ
         (void) fprintf(el->fErrFile,
                        "Returning from el_gets %d\n", num);
#endif /* DEBUG_READ */
      //break;
   }

   if ((int) cmdnum >= el->fMap.fNFunc) {              /* BUG CHECK command */
#ifdef DEBUG_EDIT
         (void) fprintf(el->fErrFile,
                        "ERROR: illegal command from key 0%o\r\n", ch);
#endif /* DEBUG_EDIT */
      //continue;	/* try again */
   }
   /* now do the real command */
#ifdef DEBUG_READ
   {
      ElBindings_t* b;

      for (b = el->fMap.fHelp; b->fName; b++) {
         if (b->fFunc == cmdnum) {
            break;
         }
      }

      if (b->fName) {
         (void) fprintf(el->fErrFile,
                        "Executing %s\n", b->fName);
      } else {
         (void) fprintf(el->fErrFile,
                        "Error command = %d\n", cmdnum);
      }
   }
#endif /* DEBUG_READ */
   retval = (*el->fMap.fFunc[cmdnum])(el, ch);

   /* save the last command here */
   el->fState.fLastCmd = cmdnum;

   if (el->fMap.fFunc[cmdnum] != ed_replay_hist) {
      el->fState.fReplayHist = -1;
   }
   /* use any return value */
   switch (retval) {
   case CC_CURSOR:
      el->fState.fArgument = 1;
      el->fState.fDoingArg = 0;
      re_refresh_cursor(el);
      break;

   case CC_REDISPLAY:
      re_clear_lines(el);
      re_clear_display(el);
   /* FALLTHROUGH */

   case CC_REFRESH:
      el->fState.fArgument = 1;
      el->fState.fDoingArg = 0;
      re_refresh(el);
      break;

   case CC_REFRESH_BEEP:
      el->fState.fArgument = 1;
      el->fState.fDoingArg = 0;
      re_refresh(el);
      term_beep(el);
      break;

   case CC_NORM:                /* normal char */
      el->fState.fArgument = 1;
      el->fState.fDoingArg = 0;
      num = el->fLine.fLastChar - el->fLine.fBuffer;                                  // LOUISE ret val
      break;

   case CC_ARGHACK:                     /* Suggested by Rich Salz */
      /* <rsalz@pineapple.bbn.com> */
      break;                    /* keep going... */

   case CC_EOF:                 /* end of file typed */
      num = 0;
      break;

   case CC_NEWLINE:                     /* normal end of line */
      num = el->fLine.fLastChar - el->fLine.fBuffer;

      if (0 == num) {
         // Added by stephan, so that
         // the readline compat layer
         // can know the difference
         // between an empty line
         // and EOF.
         el->fLine.fLastChar = el->fLine.fBuffer;
         *el->fLine.fBuffer = '\0';
         num = 1;
      }
      //printf( "el_gets() CC_NEWLINE! num=%d\n", num );
      break;

   case CC_FATAL:               /* fatal error, reset to known state */
#ifdef DEBUG_READ
         (void) fprintf(el->fErrFile,
                        "*** editor fatal ERROR ***\r\n\n");
#endif /* DEBUG_READ */
      /* put (real) cursor in a known place */
      re_clear_display(el);                     /* reset the display stuff */
      ch_reset(el);                     /* reset the input pointers */
      re_refresh(el);                   /* print the prompt again */
      el->fState.fArgument = 1;
      el->fState.fDoingArg = 0;
      break;

   case CC_ERROR:
   default:                     /* functions we don't know about */
#ifdef DEBUG_READ
         (void) fprintf(el->fErrFile,
                        "*** editor ERROR ***\r\n\n");
#endif /* DEBUG_READ */
      el->fState.fArgument = 1;
      el->fState.fDoingArg = 0;
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

   //if (el->fFlags & HANDLE_SIGNALS)
   //	sig_clr(el);
   if (nread) {
      *nread = num;
   }

   if (retval == 1) {    /* enter key pressed - add a newline */
      *el->fLine.fLastChar = '\n';
      el->fLine.fLastChar++;
      *el->fLine.fLastChar = 0;

      return el->fLine.fBuffer;
   }

   CMacro_t* ma = &el->fCharEd.fMacro;

   if (retval == CC_REFRESH && ma->fLevel >= 0 && ma->fMacro[ma->fLevel]) {
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
      if (el->fLine.fCursor <= el->fLine.fLastChar) {
         matchParentheses(el);
      }


      /* '\a' indicates entry is not complete */
      *el->fLine.fLastChar = '\a';
   }

   return num ? el->fLine.fBuffer : NULL;
} // el_gets


el_public const char*
el_gets_newline(EditLine_t* el, int* nread) {
   if (el->fFlags & HANDLE_SIGNALS) {
      sig_set(el);
   }
   re_clear_display(el);        /* reset the display stuff */

   /* '\a' is used to signal that we re-entered this function without newline being hit. */
   if (*el->fLine.fLastChar == '\a') {
      // Remove the '\a'
      // by letting getc overwrite it!
   } else {
      // Only reset the buffer if we edit a whole new line
      ch_reset(el);
      if (el->fState.fReplayHist >= 0) {
	 el->fHistory.fEventNo = el->fState.fReplayHist;
	 // load the entry
	 ed_prev_history(el, 0);
      }

   }

   if (!(el->fFlags & NO_TTY)) {
      re_refresh(el);                   /* print the prompt */
   }
   term__flush();

   if (nread) {
      *nread = 0;
   }
   return NULL;
} // el_gets_newline


el_public bool
el_eof(EditLine_t* el) {
   return !el->fLine.fBuffer[0] && !strcmp(el->fLine.fBuffer + 1, "EOF");
}


/**
   el_public const char *
   el_gets(EditLine_t *el, int *nread)
   {
        el_gets( el, nread );
 *nread = el_chop_at_newline(el);
        return el->fLine.fBuffer;
   }
 */
