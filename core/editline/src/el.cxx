// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*      $NetBSD: el.c,v 1.21 2001/01/05 22:45:30 christos Exp $ */

/*-
 * Copyright (c) 1992, 1993
 *      The Regents of the University of California.  All rights reserved.
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

#include <fstream>
#include <string>

/*
 * el.c: EditLine_t interface functions
 */
#include "sys.h"

#include <sys/types.h>
#include <sys/param.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include "el.h"

/* el_init():
 *      Initialize SEditLine_t and set default parameters.
 */
el_public EditLine_t*
el_init(const char* prog, FILE* fin, FILE* fout, FILE* ferr) {
   EditLine_t* el = (EditLine_t*) el_malloc(sizeof(EditLine_t));
#ifdef DEBUG
   char* tty;
#endif

   if (el == NULL) {
      return NULL;
   }

   memset(el, 0, sizeof(EditLine_t));

   el->fIn = fin;
   el->fInFD = fileno(fin);
   if (isatty(el->fInFD))
      el->fIn = 0;

   el->fOutFile = fout;
   el->fErrFile = ferr;
   el->fProg = strdup(prog);

   /*
    * Initialize all the modules. Order is important!!!
    */
   el->fFlags = 0;

   (void) term_init(el);
   (void) key_init(el);
   (void) map_init(el);

   if (tty_init(el) == -1) {
      el->fFlags |= NO_TTY;
   }
   (void) ch_init(el);
   (void) search_init(el);
   (void) hist_init(el);
   (void) prompt_init(el);
   (void) sig_init(el);

   return el;
} // el_init


/* el_end():
 *      Clean up.
 */
el_public void
el_end(EditLine_t* el) {
   if (el == NULL) {
      return;
   }

   el_reset(el);

   term_end(el);
   key_end(el);
   map_end(el);
   tty_end(el);
   ch_end(el);
   search_end(el);
   hist_end(el);
   prompt_end(el);
   sig_end(el);

   el_free((ptr_t) el->fProg);
   el_free((ptr_t) el);
} // el_end


/* el_reset():
 *      Reset the tty and the parser
 */
el_public void
el_reset(EditLine_t* el) {
   tty_cookedmode(el);
   ch_reset(el);                /* XXX: Do we want that? */
}


/* el_set():
 *      set the SEditLine_t parameters
 */
el_public int
el_set(EditLine_t* el, int op, ...) {
   va_list va;
   int rv;
   va_start(va, op);

   if (el == NULL) {
      va_end(va);
      return -1;
   }

   switch (op) {
   case EL_PROMPT:
   case EL_RPROMPT:
      rv = prompt_set(el, va_arg(va, ElPFunc_t), op);
      break;

   case EL_TERMINAL:
      rv = term_set(el, va_arg(va, char*));
      break;

   case EL_EDITOR:
      rv = map_set_editor(el, va_arg(va, char*));
      break;

   case EL_SIGNAL:

      if (va_arg(va, int)) {
         el->fFlags |= HANDLE_SIGNALS;
      } else {
         el->fFlags &= ~HANDLE_SIGNALS;
      }
      rv = 0;
      break;

   case EL_BIND:
   case EL_TELLTC:
   case EL_SETTC:
   case EL_ECHOTC:
   case EL_SETTY:
   {
      const char* argv[20];
      const char** cargv = 0;
      int i;

      for (i = 1; i < 20; i++) {
         if ((argv[i] = va_arg(va, const char*)) == NULL) {
            break;
         }
      }
      argv[0] = argv[1];
      cargv = argv;

      switch (op) {
      case EL_BIND:
         argv[0] = "bind";
         rv = map_bind(el, i, cargv);
         break;

      case EL_TELLTC:
         argv[0] = "telltc";
         rv = term_telltc(el, i, cargv);
         break;

      case EL_SETTC:
         argv[0] = "settc";
         rv = term_settc(el, i, cargv);
         break;

      case EL_ECHOTC:
         argv[0] = "echotc";
         rv = term_echotc(el, i, cargv);
         break;

      case EL_SETTY:
         argv[0] = "setty";
         rv = tty_stty(el, i, cargv);
         break;

      default:
         rv = -1;
         EL_ABORT((el->fErrFile, "Bad op %d\n", op));
         break;
      } // switch
      break;
   }

   case EL_ADDFN:
   {
      char* name = va_arg(va, char*);
      char* help = va_arg(va, char*);
      ElFunc_t func = va_arg(va, ElFunc_t);

      rv = map_addfunc(el, name, help, func);
      break;
   }

   case EL_HIST:
   {
      HistFun_t func = va_arg(va, HistFun_t);
      ptr_t ptr = va_arg(va, char*);

      rv = hist_set(el, func, ptr);
      break;
   }

   case EL_EDITMODE:

      if (va_arg(va, int)) {
         el->fFlags &= ~EDIT_DISABLED;
      } else {
         el->fFlags |= EDIT_DISABLED;
      }
      rv = 0;
      break;

   default:
      rv = -1;
   } // switch

   va_end(va);
   return rv;
} // el_set


/* el_get():
 *      retrieve the SEditLine_t parameters
 */
el_public int
el_get(EditLine_t* el, int op, void* ret) {
   int rv;

   if (el == NULL || ret == NULL) {
      return -1;
   }

   switch (op) {
   case EL_PROMPT:
   case EL_RPROMPT:
      {
         ElPFunc_t func;
         rv = prompt_get(el, &func, op);
         ret = (void*) func;
         break;
      }

   case EL_EDITOR:
      {
         const char* str;
         rv = map_get_editor(el, &str);
         ret = (void*)str;
         break;
      }

   case EL_SIGNAL:
      *((int*) ret) = (el->fFlags & HANDLE_SIGNALS);
      rv = 0;
      break;

   case EL_EDITMODE:
      *((int*) ret) = (!(el->fFlags & EDIT_DISABLED));
      rv = 0;
      break;

#if 0                           /* XXX */
   case EL_TERMINAL:
      rv = term_get(el, (const char*) &ret);
      break;

   case EL_BIND:
   case EL_TELLTC:
   case EL_SETTC:
   case EL_ECHOTC:
   case EL_SETTY:
   {
      char* argv[20];
      int i;

      for (i = 1; i < 20; i++) {
         if ((argv[i] = va_arg(va, char*)) == NULL) {
            break;
         }
      }

      switch (op) {
      case EL_BIND:
         argv[0] = "bind";
         rv = map_bind(el, i, argv);
         break;

      case EL_TELLTC:
         argv[0] = "telltc";
         rv = term_telltc(el, i, argv);
         break;

      case EL_SETTC:
         argv[0] = "settc";
         rv = term_settc(el, i, argv);
         break;

      case EL_ECHOTC:
         argv[0] = "echotc";
         rv = term_echotc(el, i, argv);
         break;

      case EL_SETTY:
         argv[0] = "setty";
         rv = tty_stty(el, i, argv);
         break;

      default:
         rv = -1;
         EL_ABORT((el->errfile, "Bad op %d\n", op));
         break;
      } // switch
      break;
   }

   case EL_ADDFN:
   {
      char* name = va_arg(va, char*);
      char* help = va_arg(va, char*);
      ElFunc_t func = va_arg(va, ElFunc_t);

      rv = map_addfunc(el, name, help, func);
      break;
   }

   case EL_HIST:
   {
      HistFun_t func = va_arg(va, HistFun_t);
      ptr_t ptr = va_arg(va, char*);
      rv = hist_set(el, func, ptr);
   }
   break;
#endif /* XXX */

   default:
      rv = -1;
   } // switch

   return rv;
} // el_get


/* el_line():
 *      Return editing info
 */
el_public const LineInfo_t*
el_line(EditLine_t* el) {
   return (const LineInfo_t*) (void*) &el->fLine;
}


static const char elpath[] = "/.editrc";

/* el_source():
 *      Source a file
 */
el_public int
el_source(EditLine_t* el, const char* fname) {
   char* ptr, path[MAXPATHLEN];

   if (fname == NULL) {
      if ((ptr = getenv("HOME")) == NULL) {
         return -1;
      }

      if (strlcpy(path, ptr, sizeof(path)) >= sizeof(path)) {
         return -1;
      }

      if (strlcat(path, elpath, sizeof(path)) >= sizeof(path)) {
         return -1;
      }
      fname = path;
   }

   std::ifstream in(fname);
   std::string line;
   while (in) {
      std::getline(in, line);
      if (parse_line(el, line.c_str()) == -1) {
         return -1;
      }
   }

   return 0;
} // el_source


/* el_resize():
 *      Called from program when terminal is resized
 */
el_public void
el_resize(EditLine_t* el) {
   int lins, cols;
   sigset_t oset, nset;

   (void) sigemptyset(&nset);
   (void) sigaddset(&nset, SIGWINCH);
   (void) sigprocmask(SIG_BLOCK, &nset, &oset);

   int curHPos = el->fCursor.fH;
   int curVPos = el->fCursor.fV;

   // We want to clear the old lines later. But how many did we have?
   int displen = el->fPrompt.fPos.fH;
   displen += el->fLine.fLastChar - el->fLine.fBuffer;
   // fTerm still has the old number of columns
   int nlines = displen / el->fTerm.fSize.fH;

   /* get the correct window size */
   if (term_get_size(el, &lins, &cols)) {
      term_change_size(el, lins, cols);
   }

   // Now clear the old lines.
   el->fRefresh.r_oldcv = nlines;

   // We need to set the cursor position after the resize, or refresh
   // will argue that nothing has changed (term_change_size set it to 0).
   // Set it to the last column if too large, otherwise keep it.
   el->fCursor.fH = curHPos >= cols ? cols - 1 : curHPos;
   // the vertical cursor pos does not change by resizing the window
   el->fCursor.fV = curVPos;
   re_clear_lines(el);
   re_refresh(el);
   term__flush();

   (void) sigprocmask(SIG_SETMASK, &oset, NULL);
} // el_resize


/* el_beep():
 *      Called from the program to beep
 */
el_public void
el_beep(EditLine_t* el) {
   term_beep(el);
}


/* el_editmode()
 *      Set the state of EDIT_DISABLED from the `edit' command.
 */
el_protected int
/*ARGSUSED*/
el_editmode(EditLine_t* el, int argc, const char** argv) {
   const char* how;

   if (argv == NULL || argc != 2 || argv[1] == NULL) {
      (void) fprintf(el->fErrFile, "edit: Usage error. Pass 'on' or 'off' to enable or disable line-editing mode.\n");
      return -1;
   }

   how = argv[1];

   if (strcmp(how, "on") == 0) {
      el->fFlags &= ~EDIT_DISABLED;
   } else if (strcmp(how, "off") == 0) {
      el->fFlags |= EDIT_DISABLED;
   } else {
      (void) fprintf(el->fErrFile, "edit: Bad value `%s'.\n", how);
      return -1;
   }
   return 0;
} // el_editmode
