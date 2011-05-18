// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: prompt.c,v 1.8 2001/01/10 07:45:41 jdolecek Exp $	*/

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
 * prompt.c: Prompt printing functions
 */
#include "sys.h"
#include <stdio.h>
#include "el.h"

ElColor_t prompt_color(6, -1);

el_private const char* prompt_default(EditLine_t*);
el_private const char* prompt_default_r(EditLine_t*);

/* prompt_default():
 *	Just a default prompt, in case the user did not provide one
 */
el_private const char*
/*ARGSUSED*/
prompt_default(EditLine_t* /*el*/) {
   static char a[3] = { '?', ' ', '\0' };

   return a;
}


/* prompt_default_r():
 *	Just a default rprompt, in case the user did not provide one
 */
el_private const char*
/*ARGSUSED*/
prompt_default_r(EditLine_t* /*el*/) {
   static char a[1] = { '\0' };

   return a;
}


/* prompt_print():
 *	Print the prompt and update the prompt position.
 *	We use an array of integers in case we want to pass
 *      literal escape sequences in the prompt and we want a
 *	bit to flag them
 */
el_protected void
prompt_print(EditLine_t* el, int op) {
   ElPrompt_t* elp;
   const char* p;

   if (op == EL_PROMPT) {
      elp = &el->fPrompt;
   } else {
      elp = &el->fRPrompt;
   }
   p = (elp->fFunc)(el);

   if (*p && !tty_can_output()) {
      // don't print the prompt to not block
      return;
   }


   ElColor_t col(prompt_color);

   while (*p) {
      if (*p == '\033' && p[1] == '[') {
         // escape sequence?
         // we support up to 3 numbers separated by ';'
         int num[3] = {0};
         int i = 2;
         int n = 0;
         while(n < 3) {
            while (isdigit(p[i])) {
               num[n] *= 10;
               num[n] += p[i] - '0';
               ++i;
            };
            ++n;
            if (p[i] != ';') {
               // ';' is number separator
               break;
            }
         }
         if (p[i] == 'm') {
            // color / bold / ...
            const char* strColor = 0;
            if (n < 2) {
               if (num[0] == 0) {
                  strColor = "default";
               } else if (num[0] == 1) {
                  strColor = "bold default";
               } else if (num[0] == '4') {
                  strColor = "under default";
               } else if (num[0] == '5') {
                  strColor = "bold default";
               } else if (num[0] == '7') {
                  // reverse, not supported
                  // strColor = "reverse";
               }
            } else if (num[0] == '3') {
               const char* colors[] = {
                  "black", "red", "green", "yellow", "blue",
                  "magenta" , "cyan", "white", "default"
               };
               strColor = colors[num[1]];
            } else if (num[0] == '4') {
               // bg color, not supported
            }

            if (strColor) {
               col.fForeColor = term__atocolor(strColor);
            } else {
               col.fForeColor = -1;
            }
            p += i + 1; // skip escape
            continue;
         }
      }
      re_putc(el, *p++, 1, &col);
   }

   elp->fPos.fV = el->fRefresh.r_cursor.fV;
   elp->fPos.fH = el->fRefresh.r_cursor.fH;
} // prompt_print


/* prompt_init():
 *	Initialize the prompt stuff
 */
el_protected int
prompt_init(EditLine_t* el) {
   el->fPrompt.fFunc = prompt_default;
   el->fPrompt.fPos.fV = 0;
   el->fPrompt.fPos.fH = 0;
   el->fRPrompt.fFunc = prompt_default_r;
   el->fRPrompt.fPos.fV = 0;
   el->fRPrompt.fPos.fH = 0;
   return 0;
}


/* prompt_end():
 *	Clean up the prompt stuff
 */
el_protected void
/*ARGSUSED*/
prompt_end(EditLine_t* /*el*/) {
}


/* prompt_set():
 *	Install a prompt printing function
 */
el_protected int
prompt_set(EditLine_t* el, ElPFunc_t prf, int op) {
   ElPrompt_t* p;

   if (op == EL_PROMPT) {
      p = &el->fPrompt;
   } else {
      p = &el->fRPrompt;
   }

   if (prf == NULL) {
      if (op == EL_PROMPT) {
         p->fFunc = prompt_default;
      } else {
         p->fFunc = prompt_default_r;
      }
   } else {
      p->fFunc = prf;
   }
   p->fPos.fV = 0;
   p->fPos.fH = 0;
   return 0;
} // prompt_set


/* prompt_get():
 *	Retrieve the prompt printing function
 */
el_protected int
prompt_get(EditLine_t* el, ElPFunc_t* prf, int op) {
   if (prf == NULL) {
      return -1;
   }

   if (op == EL_PROMPT) {
      *prf = el->fPrompt.fFunc;
   } else {
      *prf = el->fRPrompt.fFunc;
   }
   return 0;
}


/* prompt_setcolor():
 *	Set the prompt's dislpay color
 */
el_protected void
prompt_setcolor(int col) {
   prompt_color.fForeColor = col;
}
