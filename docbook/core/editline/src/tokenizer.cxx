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
   char* fIfs;                   /* In field separator			 */
   int fArgC;                    /* Current number of args	 */
   int fAMax;                    /* Maximum number of args	 */
   char** fArgV;                 /* Argument list			 */
   char* fWPtr;                  /* Space on the word buffer	 */
   char* fWMax;                  /* Limit on the word buffer	 */
   char* fWStart;                /* Beginning of next word		 */
   char* fWSpace;                /* Space of word buffer			 */
   Quote_t fQuote;               /* Quoting state			 */
   int fFlags;                   /* flags;				 */
};


el_private void tok_finish_word(Tokenizer_t*);


/* tok_finish_word():
 *	Finish a word in the ElTokenizer_t.
 */
el_private void
tok_finish_word(Tokenizer_t* tok) {
   *tok->fWPtr = '\0';

   if ((tok->fFlags & TOK_KEEP) || tok->fWPtr != tok->fWStart) {
      tok->fArgV[tok->fArgC++] = tok->fWStart;
      tok->fArgV[tok->fArgC] = NULL;
      tok->fWStart = ++tok->fWPtr;
   }
   tok->fFlags &= ~TOK_KEEP;
}


/* tok_init():
 *	Initialize the ElTokenizer_t
 */
el_public Tokenizer_t*
tok_init(const char* ifs) {
   Tokenizer_t* tok = (Tokenizer_t*) tok_malloc(sizeof(Tokenizer_t));

   tok->fIfs = strdup(ifs ? ifs : IFS);
   tok->fArgC = 0;
   tok->fAMax = AINCR;
   tok->fArgV = (char**) tok_malloc(sizeof(char*) * tok->fAMax);

   if (tok->fArgV == NULL) {
      tok_free((ptr_t) tok);
      return NULL;
   }
   tok->fArgV[0] = NULL;
   tok->fWSpace = (char*) tok_malloc(WINCR);

   if (tok->fWSpace == NULL) {
      tok_free((ptr_t) tok);
      return NULL;
   }
   tok->fWMax = tok->fWSpace + WINCR;
   tok->fWStart = tok->fWSpace;
   tok->fWPtr = tok->fWSpace;
   tok->fFlags = 0;
   tok->fQuote = kQuoteNone;

   return tok;
} // tok_init


/* tok_reset():
 *	Reset the ElTokenizer_t
 */
el_public void
tok_reset(Tokenizer_t* tok) {
   tok->fArgC = 0;
   tok->fWStart = tok->fWSpace;
   tok->fWPtr = tok->fWSpace;
   tok->fFlags = 0;
   tok->fQuote = kQuoteNone;
}


/* tok_end():
 *	Clean up
 */
el_public void
tok_end(Tokenizer_t* tok) {
   tok_free((ptr_t) tok->fIfs);
   tok_free((ptr_t) tok->fWSpace);
   tok_free((ptr_t) tok->fArgV);
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
         tok->fFlags |= TOK_KEEP;
         tok->fFlags &= ~TOK_EAT;

         switch (tok->fQuote) {
         case kQuoteNone:
            tok->fQuote = kQuoteSingle;                      /* Enter single quote
                                                         * mode */
            break;

         case kQuoteSingle:                 /* Exit single quote mode */
            tok->fQuote = kQuoteNone;
            break;

         case kQuoteOne:                    /* Quote this ' */
            tok->fQuote = kQuoteNone;
            *tok->fWPtr++ = *ptr;
            break;

         case kQuoteDouble:                 /* Stay in double quote mode */
            *tok->fWPtr++ = *ptr;
            break;

         case kQuoteDoubleone:                      /* Quote this ' */
            tok->fQuote = kQuoteDouble;
            *tok->fWPtr++ = *ptr;
            break;

         default:
            return -1;
         } // switch
         break;

      case '"':
         tok->fFlags &= ~TOK_EAT;
         tok->fFlags |= TOK_KEEP;

         switch (tok->fQuote) {
         case kQuoteNone:                   /* Enter double quote mode */
            tok->fQuote = kQuoteDouble;
            break;

         case kQuoteDouble:                 /* Exit double quote mode */
            tok->fQuote = kQuoteNone;
            break;

         case kQuoteOne:                    /* Quote this " */
            tok->fQuote = kQuoteNone;
            *tok->fWPtr++ = *ptr;
            break;

         case kQuoteSingle:                 /* Stay in single quote mode */
            *tok->fWPtr++ = *ptr;
            break;

         case kQuoteDoubleone:                      /* Quote this " */
            tok->fQuote = kQuoteDouble;
            *tok->fWPtr++ = *ptr;
            break;

         default:
            return -1;
         } // switch
         break;

      case '\\':
         tok->fFlags |= TOK_KEEP;
         tok->fFlags &= ~TOK_EAT;

         switch (tok->fQuote) {
         case kQuoteNone:                   /* Quote next character */
            tok->fQuote = kQuoteOne;
            break;

         case kQuoteDouble:                 /* Quote next character */
            tok->fQuote = kQuoteDoubleone;
            break;

         case kQuoteOne:                    /* Quote this, restore state */
            *tok->fWPtr++ = *ptr;
            tok->fQuote = kQuoteNone;
            break;

         case kQuoteSingle:                 /* Stay in single quote mode */
            *tok->fWPtr++ = *ptr;
            break;

         case kQuoteDoubleone:                      /* Quote this \ */
            tok->fQuote = kQuoteDouble;
            *tok->fWPtr++ = *ptr;
            break;

         default:
            return -1;
         } // switch
         break;

      case '\n':
         tok->fFlags &= ~TOK_EAT;

         switch (tok->fQuote) {
         case kQuoteNone:
            tok_finish_word(tok);
            *argv = tok->fArgV;
            *argc = tok->fArgC;
            return 0;

         case kQuoteSingle:
         case kQuoteDouble:
            *tok->fWPtr++ = *ptr;                        /* Add the return */
            break;

         case kQuoteDoubleone:                  /* Back to double, eat the '\n' */
            tok->fFlags |= TOK_EAT;
            tok->fQuote = kQuoteDouble;
            break;

         case kQuoteOne:                    /* No quote, more eat the '\n' */
            tok->fFlags |= TOK_EAT;
            tok->fQuote = kQuoteNone;
            break;

         default:
            return 0;
         } // switch
         break;

      case '\0':

         switch (tok->fQuote) {
         case kQuoteNone:

            /* Finish word and return */
            if (tok->fFlags & TOK_EAT) {
               tok->fFlags &= ~TOK_EAT;
               return 3;
            }
            tok_finish_word(tok);
            *argv = tok->fArgV;
            *argc = tok->fArgC;
            return 0;

         case kQuoteSingle:
            return 1;

         case kQuoteDouble:
            return 2;

         case kQuoteDoubleone:
            tok->fQuote = kQuoteDouble;
            *tok->fWPtr++ = *ptr;
            break;

         case kQuoteOne:
            tok->fQuote = kQuoteNone;
            *tok->fWPtr++ = *ptr;
            break;

         default:
            return -1;
         } // switch
         break;

      default:
         tok->fFlags &= ~TOK_EAT;

         switch (tok->fQuote) {
         case kQuoteNone:

            if (strchr(tok->fIfs, *ptr) != NULL) {
               tok_finish_word(tok);
            } else {
               *tok->fWPtr++ = *ptr;
            }
            break;

         case kQuoteSingle:
         case kQuoteDouble:
            *tok->fWPtr++ = *ptr;
            break;


         case kQuoteDoubleone:
            *tok->fWPtr++ = '\\';
            tok->fQuote = kQuoteDouble;
            *tok->fWPtr++ = *ptr;
            break;

         case kQuoteOne:
            tok->fQuote = kQuoteNone;
            *tok->fWPtr++ = *ptr;
            break;

         default:
            return -1;

         } // switch
         break;
      } // switch

      if (tok->fWPtr >= tok->fWMax - 4) {
         size_t size = tok->fWMax - tok->fWSpace + WINCR;
         char* s = (char*) tok_realloc(tok->fWSpace, size);
         /* SUPPRESS 22 */
         int offs = s - tok->fWSpace;

         if (s == NULL) {
            return -1;
         }

         if (offs != 0) {
            int i;

            for (i = 0; i < tok->fArgC; i++) {
               tok->fArgV[i] = tok->fArgV[i] + offs;
            }
            tok->fWPtr = tok->fWPtr + offs;
            tok->fWStart = tok->fWStart + offs;
            tok->fWMax = s + size;
            tok->fWSpace = s;
         } else {
            tok_free((ptr_t) s);
         }
      }

      if (tok->fArgC >= tok->fAMax - 4) {
         char** p;
         tok->fAMax += AINCR;
         p = (char**) tok_realloc(tok->fArgV,
                                  tok->fAMax * sizeof(char*));

         if (p == NULL) {
            return -1;
         }
         tok->fArgV = p;
      }
   }
   return 0;      /* ??? added by stephan */
} // tok_line
