// @(#)root/editline:$Id$
// Author: Mary-Louise Gill, 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*	$NetBSD: ElTokenizer_t.c,v 1.7 2001/01/04 15:56:32 christos Exp $	*/

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
 * tokenize.c: Bourne shell like ElTokenizer_t
 */
#include "sys.h"
#include <string.h>
#include <stdlib.h>
#include "tokenizer.h"

typedef enum {
   kQuoteNone, kQuoteSingle, kQuoteDouble, kQuoteOne, kQuoteDoubleone
} Quote_t;

#define IFS "\t \n"

#define TOK_KEEP 1
#define TOK_EAT 2

#define WINCR 20
#define AINCR 10

#define tok_malloc(a) malloc(a)
#define tok_free(a) free(a)
#define tok_realloc(a, b) realloc(a, b)


struct ElTokenizer_t {
   char* ifs;                   /* In field separator			 */
   int argc, amax;              /* Current and maximum number of args	 */
   char** argv;                 /* Argument list			 */
   char* wptr, * wmax;          /* Space and limit on the word buffer	 */
   char* wstart;                /* Beginning of next word		 */
   char* wspace;                /* Space of word buffer			 */
   Quote_t quote;               /* Quoting state			 */
   int flags;                   /* flags;				 */
};


el_private void tok_finish_word(Tokenizer_t*);


/* tok_finish_word():
 *	Finish a word in the ElTokenizer_t.
 */
el_private void
tok_finish_word(Tokenizer_t* tok) {
   *tok->wptr = '\0';

   if ((tok->flags & TOK_KEEP) || tok->wptr != tok->wstart) {
      tok->argv[tok->argc++] = tok->wstart;
      tok->argv[tok->argc] = NULL;
      tok->wstart = ++tok->wptr;
   }
   tok->flags &= ~TOK_KEEP;
}


/* tok_init():
 *	Initialize the ElTokenizer_t
 */
el_public Tokenizer_t*
tok_init(const char* ifs) {
   Tokenizer_t* tok = (Tokenizer_t*) tok_malloc(sizeof(Tokenizer_t));

   tok->ifs = strdup(ifs ? ifs : IFS);
   tok->argc = 0;
   tok->amax = AINCR;
   tok->argv = (char**) tok_malloc(sizeof(char*) * tok->amax);

   if (tok->argv == NULL) {
      return NULL;
   }
   tok->argv[0] = NULL;
   tok->wspace = (char*) tok_malloc(WINCR);

   if (tok->wspace == NULL) {
      return NULL;
   }
   tok->wmax = tok->wspace + WINCR;
   tok->wstart = tok->wspace;
   tok->wptr = tok->wspace;
   tok->flags = 0;
   tok->quote = kQuoteNone;

   return tok;
} // tok_init


/* tok_reset():
 *	Reset the ElTokenizer_t
 */
el_public void
tok_reset(Tokenizer_t* tok) {
   tok->argc = 0;
   tok->wstart = tok->wspace;
   tok->wptr = tok->wspace;
   tok->flags = 0;
   tok->quote = kQuoteNone;
}


/* tok_end():
 *	Clean up
 */
el_public void
tok_end(Tokenizer_t* tok) {
   tok_free((ptr_t) tok->ifs);
   tok_free((ptr_t) tok->wspace);
   tok_free((ptr_t) tok->argv);
   tok_free((ptr_t) tok);
}


/* tok_line():
 *	Bourne shell like tokenizing
 *	Return:
 *		-1: Internal error
 *		 3: Quoted return
 *		 2: Unmatched double quote
 *		 1: Unmatched single quote
 *		 0: Ok
 */
el_public int
tok_line(Tokenizer_t* tok, const char* line, int* argc, char*** argv) {
   const char* ptr;

   for ( ; ;) {
      switch (*(ptr = line++)) {
      case '\'':
         tok->flags |= TOK_KEEP;
         tok->flags &= ~TOK_EAT;

         switch (tok->quote) {
         case kQuoteNone:
            tok->quote = kQuoteSingle;                      /* Enter single quote
                                                         * mode */
            break;

         case kQuoteSingle:                 /* Exit single quote mode */
            tok->quote = kQuoteNone;
            break;

         case kQuoteOne:                    /* Quote this ' */
            tok->quote = kQuoteNone;
            *tok->wptr++ = *ptr;
            break;

         case kQuoteDouble:                 /* Stay in double quote mode */
            *tok->wptr++ = *ptr;
            break;

         case kQuoteDoubleone:                      /* Quote this ' */
            tok->quote = kQuoteDouble;
            *tok->wptr++ = *ptr;
            break;

         default:
            return -1;
         } // switch
         break;

      case '"':
         tok->flags &= ~TOK_EAT;
         tok->flags |= TOK_KEEP;

         switch (tok->quote) {
         case kQuoteNone:                   /* Enter double quote mode */
            tok->quote = kQuoteDouble;
            break;

         case kQuoteDouble:                 /* Exit double quote mode */
            tok->quote = kQuoteNone;
            break;

         case kQuoteOne:                    /* Quote this " */
            tok->quote = kQuoteNone;
            *tok->wptr++ = *ptr;
            break;

         case kQuoteSingle:                 /* Stay in single quote mode */
            *tok->wptr++ = *ptr;
            break;

         case kQuoteDoubleone:                      /* Quote this " */
            tok->quote = kQuoteDouble;
            *tok->wptr++ = *ptr;
            break;

         default:
            return -1;
         } // switch
         break;

      case '\\':
         tok->flags |= TOK_KEEP;
         tok->flags &= ~TOK_EAT;

         switch (tok->quote) {
         case kQuoteNone:                   /* Quote next character */
            tok->quote = kQuoteOne;
            break;

         case kQuoteDouble:                 /* Quote next character */
            tok->quote = kQuoteDoubleone;
            break;

         case kQuoteOne:                    /* Quote this, restore state */
            *tok->wptr++ = *ptr;
            tok->quote = kQuoteNone;
            break;

         case kQuoteSingle:                 /* Stay in single quote mode */
            *tok->wptr++ = *ptr;
            break;

         case kQuoteDoubleone:                      /* Quote this \ */
            tok->quote = kQuoteDouble;
            *tok->wptr++ = *ptr;
            break;

         default:
            return -1;
         } // switch
         break;

      case '\n':
         tok->flags &= ~TOK_EAT;

         switch (tok->quote) {
         case kQuoteNone:
            tok_finish_word(tok);
            *argv = tok->argv;
            *argc = tok->argc;
            return 0;

         case kQuoteSingle:
         case kQuoteDouble:
            *tok->wptr++ = *ptr;                        /* Add the return */
            break;

         case kQuoteDoubleone:                  /* Back to double, eat the '\n' */
            tok->flags |= TOK_EAT;
            tok->quote = kQuoteDouble;
            break;

         case kQuoteOne:                    /* No quote, more eat the '\n' */
            tok->flags |= TOK_EAT;
            tok->quote = kQuoteNone;
            break;

         default:
            return 0;
         } // switch
         break;

      case '\0':

         switch (tok->quote) {
         case kQuoteNone:

            /* Finish word and return */
            if (tok->flags & TOK_EAT) {
               tok->flags &= ~TOK_EAT;
               return 3;
            }
            tok_finish_word(tok);
            *argv = tok->argv;
            *argc = tok->argc;
            return 0;

         case kQuoteSingle:
            return 1;

         case kQuoteDouble:
            return 2;

         case kQuoteDoubleone:
            tok->quote = kQuoteDouble;
            *tok->wptr++ = *ptr;
            break;

         case kQuoteOne:
            tok->quote = kQuoteNone;
            *tok->wptr++ = *ptr;
            break;

         default:
            return -1;
         } // switch
         break;

      default:
         tok->flags &= ~TOK_EAT;

         switch (tok->quote) {
         case kQuoteNone:

            if (strchr(tok->ifs, *ptr) != NULL) {
               tok_finish_word(tok);
            } else {
               *tok->wptr++ = *ptr;
            }
            break;

         case kQuoteSingle:
         case kQuoteDouble:
            *tok->wptr++ = *ptr;
            break;


         case kQuoteDoubleone:
            *tok->wptr++ = '\\';
            tok->quote = kQuoteDouble;
            *tok->wptr++ = *ptr;
            break;

         case kQuoteOne:
            tok->quote = kQuoteNone;
            *tok->wptr++ = *ptr;
            break;

         default:
            return -1;

         } // switch
         break;
      } // switch

      if (tok->wptr >= tok->wmax - 4) {
         size_t size = tok->wmax - tok->wspace + WINCR;
         char* s = (char*) tok_realloc(tok->wspace, size);
         /* SUPPRESS 22 */
         int offs = s - tok->wspace;

         if (s == NULL) {
            return -1;
         }

         if (offs != 0) {
            int i;

            for (i = 0; i < tok->argc; i++) {
               tok->argv[i] = tok->argv[i] + offs;
            }
            tok->wptr = tok->wptr + offs;
            tok->wstart = tok->wstart + offs;
            tok->wmax = s + size;
            tok->wspace = s;
         }
      }

      if (tok->argc >= tok->amax - 4) {
         char** p;
         tok->amax += AINCR;
         p = (char**) tok_realloc(tok->argv,
                                  tok->amax * sizeof(char*));

         if (p == NULL) {
            return -1;
         }
         tok->argv = p;
      }
   }
   return 0;      /* ??? added by stephan */
} // tok_line
