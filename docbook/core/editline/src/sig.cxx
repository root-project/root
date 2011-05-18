// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: sig.c,v 1.8 2001/01/09 17:31:04 jdolecek Exp $	*/

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
 * sig.c: Signal handling stuff.
 *	  our policy is to trap all signals, set a good state
 *	  and pass the ball to our caller.
 */
#include "sys.h"
#include "el.h"
#include <stdlib.h>

el_private EditLine_t* sel = NULL;

el_private const int sighdl[] = {
#define _DO(a) (a),
   ALLSIGS
#undef  _DO
   - 1
};

el_private extern "C" void sig_handler(int);

/* sig_handler():
 *	This is the handler called for all signals
 *	XXX: we cannot pass any data so we just store the old SEditLine_t
 *	state in a el_private variable
 */
el_private void
sig_handler(int signo) {

   sigset_t nset, oset;
   (void) sigemptyset(&nset);
   (void) sigaddset(&nset, signo);
   /* not needed; a signal is always blocked before invoking
      the signal handler for that signal.
   (void) sigprocmask(SIG_BLOCK, &nset, &oset);
   */

   switch (signo) {
   case SIGCONT:
      if (tty_can_output()) {
         tty_rawmode(sel);
         //if (ed_redisplay(sel, 0) == CC_REFRESH) {
         re_clear_display(sel);
         re_refresh(sel);
         //}
         term__flush();
      }
      break;

   case SIGWINCH:
      el_resize(sel);
      break;

   default:
      tty_cookedmode(sel);
      break;
   } // switch

   int i = 0;
   for (; sighdl[i] != -1; i++) {
      if (signo == sighdl[i]) {
         break;
      }
   }
   if (sighdl[i] == -1)
      i = -1;
   else {
      (void) sigprocmask(SIG_UNBLOCK, &nset, &oset);
      (void) signal(signo, sel->fSignal[i]);
      // forward to previous signal handler:
      (void) kill(0, signo);
      (void) sigprocmask(SIG_SETMASK, &oset, NULL);

      // re-enable us
      sig_t s;
      /* This could happen if we get interrupted */
      if (i != -1 ) {
         if ((s = signal(signo, sig_handler)) != sig_handler) {
            sel->fSignal[i] = s;
         }
      }
   }
} // sig_handler


/* sig_init():
 *	Initialize all signal stuff
 */
el_protected int
sig_init(EditLine_t* el) {
   int i;
   sigset_t nset, oset;

   (void) sigemptyset(&nset);
#define _DO(a) (void) sigaddset(&nset, a);
   ALLSIGS
#undef  _DO
      (void) sigprocmask(SIG_BLOCK, &nset, &oset);

#define SIGSIZE (sizeof(sighdl) / sizeof(sighdl[0]) * sizeof(sig_t))

   el->fSignal = (sig_t*) el_malloc(SIGSIZE);

   if (el->fSignal == NULL) {
      return -1;
   }

   for (i = 0; sighdl[i] != -1; i++) {
      el->fSignal[i] = SIG_ERR;
   }

   (void) sigprocmask(SIG_SETMASK, &oset, NULL);

   return 0;
} // sig_init


/* sig_end():
 *	Clear all signal stuff
 */
el_protected void
sig_end(EditLine_t* el) {
   el_free((ptr_t) el->fSignal);
   el->fSignal = NULL;
}


/* sig_set():
 *	set all the signal handlers
 */
el_protected void
sig_set(EditLine_t* el) {
   int i;
   sigset_t nset, oset;

   (void) sigemptyset(&nset);
#define _DO(a) (void) sigaddset(&nset, a);
   ALLSIGS
#undef  _DO
      (void) sigprocmask(SIG_BLOCK, &nset, &oset);

   for (i = 0; sighdl[i] != -1; i++) {
      sig_t s;

      /* This could happen if we get interrupted */
      if ((s = signal(sighdl[i], sig_handler)) != sig_handler) {
         el->fSignal[i] = s;
      }
   }
   sel = el;
   (void) sigprocmask(SIG_SETMASK, &oset, NULL);
} // sig_set


/* sig_clr():
 *	clear all the signal handlers
 */
el_protected void
sig_clr(EditLine_t* el) {
   int i;
   sigset_t nset, oset;

   (void) sigemptyset(&nset);
#define _DO(a) (void) sigaddset(&nset, a);
   ALLSIGS
#undef  _DO
      (void) sigprocmask(SIG_BLOCK, &nset, &oset);

   for (i = 0; sighdl[i] != -1; i++) {
      if (el->fSignal[i] != SIG_ERR) {
         (void) signal(sighdl[i], el->fSignal[i]);
      }
   }

   sel = NULL;                  /* we are going to die if the handler is
                                 * called */
   (void) sigprocmask(SIG_SETMASK, &oset, NULL);
} // sig_clr
