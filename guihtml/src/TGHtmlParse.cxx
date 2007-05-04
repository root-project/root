// $Id: TGHtmlParse.cxx,v 1.1 2007/05/04 17:07:01 brun Exp $
// Author:  Valeriy Onuchin   03/05/2007

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**************************************************************************

    HTML widget for xclass. Based on tkhtml 1.28
    Copyright (C) 1997-2000 D. Richard Hipp <drh@acm.org>
    Copyright (C) 2002-2003 Hector Peraza.

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

**************************************************************************/

// A tokenizer that converts raw HTML into a linked list of HTML elements.

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>

#include "TGHtml.h"
#include "TGHtmlTokens.h"


//----------------------------------------------------------------------

extern SHtmlTokenMap HtmlMarkupMap[];


/****************** Begin Escape Sequence Translator *************/

// The next section of code implements routines used to translate
// the '&' escape sequences of SGML to individual characters.
// Examples:
//
//         &amp;          &
//         &lt;           <
//         &gt;           >
//         &nbsp;         nonbreakable space
//

// Each escape sequence is recorded as an instance of the following
// structure

struct sgEsc {
   char *zName;            // The name of this escape sequence.  ex:  "amp"
   char  value[8];         // The value for this sequence.       ex:  "&"
   sgEsc *pNext;           // Next sequence with the same hash on zName
};

// The following is a table of all escape sequences.  Add new sequences
// by adding entries to this table.

static struct sgEsc esc_sequences[] = {
  { "quot",      "\"",    0 },
  { "amp",       "&",     0 },
  { "lt",        "<",     0 },
  { "gt",        ">",     0 },
  { "nbsp",      " ",     0 },
  { "iexcl",     "\241",  0 },
  { "cent",      "\242",  0 },
  { "pound",     "\243",  0 },
  { "curren",    "\244",  0 },
  { "yen",       "\245",  0 },
  { "brvbar",    "\246",  0 },
  { "sect",      "\247",  0 },
  { "uml",       "\250",  0 },
  { "copy",      "\251",  0 },
  { "ordf",      "\252",  0 },
  { "laquo",     "\253",  0 },
  { "not",       "\254",  0 },
  { "shy",       "\255",  0 },
  { "reg",       "\256",  0 },
  { "macr",      "\257",  0 },
  { "deg",       "\260",  0 },
  { "plusmn",    "\261",  0 },
  { "sup2",      "\262",  0 },
  { "sup3",      "\263",  0 },
  { "acute",     "\264",  0 },
  { "micro",     "\265",  0 },
  { "para",      "\266",  0 },
  { "middot",    "\267",  0 },
  { "cedil",     "\270",  0 },
  { "sup1",      "\271",  0 },
  { "ordm",      "\272",  0 },
  { "raquo",     "\273",  0 },
  { "frac14",    "\274",  0 },
  { "frac12",    "\275",  0 },
  { "frac34",    "\276",  0 },
  { "iquest",    "\277",  0 },
  { "Agrave",    "\300",  0 },
  { "Aacute",    "\301",  0 },
  { "Acirc",     "\302",  0 },
  { "Atilde",    "\303",  0 },
  { "Auml",      "\304",  0 },
  { "Aring",     "\305",  0 },
  { "AElig",     "\306",  0 },
  { "Ccedil",    "\307",  0 },
  { "Egrave",    "\310",  0 },
  { "Eacute",    "\311",  0 },
  { "Ecirc",     "\312",  0 },
  { "Euml",      "\313",  0 },
  { "Igrave",    "\314",  0 },
  { "Iacute",    "\315",  0 },
  { "Icirc",     "\316",  0 },
  { "Iuml",      "\317",  0 },
  { "ETH",       "\320",  0 },
  { "Ntilde",    "\321",  0 },
  { "Ograve",    "\322",  0 },
  { "Oacute",    "\323",  0 },
  { "Ocirc",     "\324",  0 },
  { "Otilde",    "\325",  0 },
  { "Ouml",      "\326",  0 },
  { "times",     "\327",  0 },
  { "Oslash",    "\330",  0 },
  { "Ugrave",    "\331",  0 },
  { "Uacute",    "\332",  0 },
  { "Ucirc",     "\333",  0 },
  { "Uuml",      "\334",  0 },
  { "Yacute",    "\335",  0 },
  { "THORN",     "\336",  0 },
  { "szlig",     "\337",  0 },
  { "agrave",    "\340",  0 },
  { "aacute",    "\341",  0 },
  { "acirc",     "\342",  0 },
  { "atilde",    "\343",  0 },
  { "auml",      "\344",  0 },
  { "aring",     "\345",  0 },
  { "aelig",     "\346",  0 },
  { "ccedil",    "\347",  0 },
  { "egrave",    "\350",  0 },
  { "eacute",    "\351",  0 },
  { "ecirc",     "\352",  0 },
  { "euml",      "\353",  0 },
  { "igrave",    "\354",  0 },
  { "iacute",    "\355",  0 },
  { "icirc",     "\356",  0 },
  { "iuml",      "\357",  0 },
  { "eth",       "\360",  0 },
  { "ntilde",    "\361",  0 },
  { "ograve",    "\362",  0 },
  { "oacute",    "\363",  0 },
  { "ocirc",     "\364",  0 },
  { "otilde",    "\365",  0 },
  { "ouml",      "\366",  0 },
  { "divide",    "\367",  0 },
  { "oslash",    "\370",  0 },
  { "ugrave",    "\371",  0 },
  { "uacute",    "\372",  0 },
  { "ucirc",     "\373",  0 },
  { "uuml",      "\374",  0 },
  { "yacute",    "\375",  0 },
  { "thorn",     "\376",  0 },
  { "yuml",      "\377",  0 },
};


// The size of the handler hash table.  For best results this should
// be a prime number which is about the same size as the number of
// escape sequences known to the system.

#define ESC_HASH_SIZE (sizeof(esc_sequences)/sizeof(esc_sequences[0])+7)


// The hash table 
//
// If the name of an escape sequence hashes to the value H, then
// apEscHash[H] will point to a linked list of Esc structures, one of
// which will be the Esc structure for that escape sequence.

static struct sgEsc *apEscHash[ESC_HASH_SIZE];


// Hash a escape sequence name. The value returned is an integer
// between 0 and ESC_HASH_SIZE-1, inclusive.

static int EscHash(const char *zName) {
  int h = 0;      // The hash value to be returned
  char c;         // The next character in the name being hashed

  while ((c = *zName) != 0) {
    h = h<<5 ^ h ^ c;
    zName++;
  }
  if (h < 0) h = -h;

  return h % ESC_HASH_SIZE;
}

#ifdef TEST
// Compute the longest and average collision chain length for the
// escape sequence hash table

static void EscHashStats() {
  int i;
  int sum = 0;
  int max = 0;
  int cnt;
  int notempty = 0;
  struct sgEsc *p;

  for (i = 0; i < sizeof(esc_sequences) / sizeof(esc_sequences[0]); i++) {
    cnt = 0; 
    p = apEscHash[i];
    if (p) notempty++;
    while (p) {
      ++cnt;
      p = p->pNext;
    }
    sum += cnt;
    if (cnt > max) max = cnt;
  }
  printf("Longest chain=%d  avg=%g  slots=%d  empty=%d (%g%%)\n",
     max, (double)sum/(double)notempty, i, i-notempty, 
     100.0*(i-notempty)/(double)i);
}
#endif

// Initialize the escape sequence hash table

static void EscInit() {
  int i;  /* For looping thru the list of escape sequences */
  int h;  /* The hash on a sequence */

  for (i = 0; i < int(sizeof(esc_sequences) / sizeof(esc_sequences[i])); i++) {
/* #ifdef XCLASS_UTF_MAX */
#if 0
    {
      int c = esc_sequences[i].value[0];
      xclass::UniCharToUtf(c, esc_sequences[i].value);
    }
#endif
    h = EscHash(esc_sequences[i].zName);
    esc_sequences[i].pNext = apEscHash[h];
    apEscHash[h] = &esc_sequences[i];
  }
#ifdef TEST
  EscHashStats();
#endif
}


// This table translates the non-standard microsoft characters between 0x80
// and 0x9f into plain ASCII so that the characters will be visible on Unix
// systems. Care is taken to translate the characters into values less than
// 0x80, to avoid UTF-8 problems.

static char acMsChar[] = {
   /* 0x80 */ 'C',
   /* 0x81 */ ' ',
   /* 0x82 */ ',',
   /* 0x83 */ 'f',
   /* 0x84 */ '"',
   /* 0x85 */ '.',
   /* 0x86 */ '*',
   /* 0x87 */ '*',
   /* 0x88 */ '^',
    /* 0x89 */ '%',
   /* 0x8a */ 'S',
   /* 0x8b */ '<',
   /* 0x8c */ 'O',
   /* 0x8d */ ' ',
   /* 0x8e */ 'Z',
   /* 0x8f */ ' ',
   /* 0x90 */ ' ',
   /* 0x91 */ '\'',
   /* 0x92 */ '\'',
   /* 0x93 */ '"',
   /* 0x94 */ '"',
   /* 0x95 */ '*',
   /* 0x96 */ '-',
   /* 0x97 */ '-',
   /* 0x98 */ '~',
   /* 0x99 */ '@',
   /* 0x9a */ 's',
   /* 0x9b */ '>',
   /* 0x9c */ 'o',
   /* 0x9d */ ' ',
   /* 0x9e */ 'z',
   /* 0x9f */ 'Y',
};


//______________________________________________________________________________
void HtmlTranslateEscapes(char *z)
{
   // Translate escape sequences in the string "z".  "z" is overwritten
   // with the translated sequence.
   //
   // Unrecognized escape sequences are unaltered.
   //
   // Example:
   //
   //      input  = "AT&amp;T &gt MCI"
   //      output = "AT&T > MCI"

   int from;   // Read characters from this position in z[]
   int to;     // Write characters into this position in z[]
   int h;      // A hash on the escape sequence
   struct sgEsc *p;  // For looping down the escape sequence collision chain
   static int isInit = 0;   // True after initialization

   from = to = 0;
   if (!isInit) {
      EscInit();
      isInit = 1;
   }
   while (z[from]) {
      if (z[from] == '&') {
         if (z[from+1] == '#') {
            int i = from + 2;
            int v = 0;
            while (isdigit(z[i])) {
               v = v*10 + z[i] - '0';
               i++;
            }
            if (z[i] == ';') { i++; }

            // Translate the non-standard microsoft characters in the range of
            // 0x80 to 0x9f into something we can see.

            if (v >= 0x80 && v < 0xa0) {
               v = acMsChar[v & 0x1f];
            }

            // Put the character in the output stream in place of the "&#000;". 
            // How we do this depends on whether or not we are using UTF-8.

            z[to++] = v;
            from = i;
         } else {
            int i = from+1;
            int c;
            while (z[i] && isalnum(z[i])) ++i;
            c = z[i];
            z[i] = 0;
            h = EscHash(&z[from+1]);
            p = apEscHash[h];
            while (p && strcmp(p->zName, &z[from+1]) != 0) p = p->pNext; 
            z[i] = c;
            if (p) {
               int j;
               for (j = 0; p->value[j]; ++j) z[to++] = p->value[j];
               from = i;
               if (c == ';') from++;
            } else {
               z[to++] = z[from++];
            }
         }

         // Look for the non-standard microsoft characters between 0x80 and 0x9f
         // and translate them into printable ASCII codes. Separate algorithms
         // are required to do this for plain ascii and for utf-8.

      } else if (((unsigned char) z[from]) >= 0x80 &&
                 ((unsigned char) z[from]) < 0xa0) {
         z[to++] = acMsChar[z[from++] & 0x1f];
      } else {
         z[to++] = z[from++];
      }
   }
   z[to] = 0;
}

/******************* End Escape Sequence Translator ***************/

/******************* Begin HTML tokenizer code *******************/

// The following variable becomes TRUE when the markup hash table
// (stored in HtmlMarkupMap[]) is initialized.

static int isInit = 0;

// The hash table for HTML markup names.
//
// If an HTML markup name hashes to H, then apMap[H] will point to
// a linked list of sgMap structure, one of which will describe the
// the particular markup (if it exists.)

static SHtmlTokenMap *apMap[HTML_MARKUP_HASH_SIZE];

// Hash a markup name
//
// HTML markup is case insensitive, so this function will give the
// same hash regardless of the case of the markup name.
//
// The value returned is an integer between 0 and HTML_MARKUP_HASH_SIZE-1,
// inclusive.

static int HtmlHash(const char *zName) {
  int h = 0;
  char c;

  while ((c = *zName) != 0) {
    if (isupper(c)) {   // do we have to check for this??????
      c = tolower(c);
    }
    h = h<<5 ^ h ^ c;
    zName++;
  }
  if (h < 0) {
    h = -h;
  }

  return h % HTML_MARKUP_HASH_SIZE;
}


#ifdef TEST
// Compute the longest and average collision chain length for the
// markup hash table

static void HtmlHashStats() {
  int i;
  int sum = 0;
  int max = 0;
  int cnt;
  int notempty = 0;
  struct sgMap *p;

   for (i = 0; i < HTML_MARKUP_COUNT; i++) {
      cnt = 0; 
      p = apMap[i];
      if (p) notempty++;
      while (p) {
         cnt++;
         p = p->pCollide;
      }
      sum += cnt;
      if (cnt > max) max = cnt;
  }

  printf("longest chain=%d  avg=%g  slots=%d  empty=%d (%g%%)\n",
         max, (double)sum/(double)notempty, i, i-notempty,
         100.0*(i-notempty)/(double)i);
}
#endif


// Initialize the escape sequence hash table

static void HtmlHashInit(void){
  int i;
  int h;  // The hash on a markup name

  for (i = 0; i < HTML_MARKUP_COUNT; i++) {
    h = HtmlHash(HtmlMarkupMap[i].zName);
    HtmlMarkupMap[i].pCollide = apMap[h];
    apMap[h] = &HtmlMarkupMap[i];
  }
#ifdef TEST
  HtmlHashStats();
#endif
}

//______________________________________________________________________________
void TGHtml::AppendElement(TGHtmlElement *pElem)
{
   // Append the given TGHtmlElement to the tokenizers list of elements

   pElem->pNext = 0;
   pElem->pPrev = pLast;
   if (pFirst == 0) {
      pFirst = pElem;
   } else {
      pLast->pNext = pElem;
   }
   pLast = pElem;
   nToken++;
}

//______________________________________________________________________________
void TGHtml::AppToken(TGHtmlElement *pNew, TGHtmlElement *p, int offs)
{
   // Insert token pNew before token p

   if (offs < 0) {
      if (p) {
         offs = p->offs;
      } else {
         offs = nText;
      }
   }

////if (p) { pNew->style = p->style; pNew->flags = p->flags; }

//  pNew->count = 0;
   pNew->offs = offs;
   pNew->pNext = p;
   if (p) {
      pNew->id = p->id;
      p->id = ++idind;
      pNew->pPrev = p->pPrev;
      if (p->pPrev) p->pPrev->pNext = pNew;
      if (pFirst == p) pFirst = pNew;
      p->pPrev = pNew;
   } else {
      pNew->id = ++idind;
      AppendElement(pNew);
   }
   nToken++;
}

//______________________________________________________________________________
static int NextColumn(int iCol, char c)
{
   // Compute the new column index following the given character.

   switch (c) {
      case '\n': return 0;
      case '\t': return (iCol | 7) + 1;
      default:   return iCol+1;
   }
   /* NOT REACHED */
}

//______________________________________________________________________________
void ToLower(char *z)
{
   // Convert a string to all lower-case letters.

   while (*z) {
      if (isupper(*z)) *z = tolower(*z);
      z++;
   }
}

//______________________________________________________________________________
int TGHtml::Tokenize()
{
   // Process as much of the input HTML as possible. Construct new
   // TGHtmlElement objects and appended them to the list. Return
   // the number of characters actually processed.
   //
   // This routine may invoke a callback procedure which could delete
   // the HTML widget. 
   //
   // This routine is not reentrant for the same HTML widget.  To
   // prevent reentrancy (during a callback), the p->iCol field is
   // set to a negative number. This is a flag to future invocations
   // not to reentry this routine. The p->iCol field is restored
   // before exiting, of course.

   char *z;             // The input HTML text
   int c;               // The next character of input
   int n;               // Number of characters processed so far
   int inpCol;          // Column of input
   int i, j;            // Loop counters
   int h;               // Result from HtmlHash()
   TGHtmlElement *pElem; // A new HTML element
   int selfClose;       // True for content free elements. Ex: <br/>
   int argc;            // The number of arguments on a markup
   SHtmlTokenMap *pMap; // For searching the markup name hash table
# define mxARG 200     // Maximum number of parameters in a single markup
   char *argv[mxARG];   // Pointers to each markup argument.
   int arglen[mxARG];   // Length of each markup argument
   //int rl, ol;
   int pIsInScript = 0;
   int pIsInNoScript = 0;
   int pIsInNoFrames = 0;
   int sawdot = 0;
   int inLi = 0;

   inpCol = iCol;
   n = nComplete;
   z = zText;
   if (inpCol < 0) return n;   // Prevents recursion
   iCol = -1;
   pElem = 0;

   while ((c = z[n]) != 0) {

      sawdot--;
      if (c == -64 && z[n+1] == -128) { 
         n += 2;
         continue;
      }

      if (pScript) {

         // We are in the middle of <SCRIPT>...</SCRIPT>.  Just look for
         // the </SCRIPT> markup.  (later:)  Treat <STYLE>...</STYLE> the
         // same way.

         TGHtmlScript *pScr = pScript;
         char *zEnd;
         int nEnd;
         //int curline, curch, curlast = n;
         int sqcnt;
         if (pScr->type == Html_SCRIPT) {
            zEnd = "</script>";
            nEnd = 9;
         } else if (pScr->type == Html_NOSCRIPT) {
            zEnd = "</noscript>";
            nEnd = 11;
         } else if (pScr->type == Html_NOFRAMES) {
            zEnd = "</noframes>";
            nEnd = 11;
         } else {
            zEnd = "</style>";
            nEnd = 8;
         }
         if (pScr->nStart < 0) {
            pScr->nStart = n;
            pScr->nScript = 0;
         }
         sqcnt = 0;
         for (i = n /*pScr->nStart + pScr->nScript*/; z[i]; i++) {
            if (z[i] == '\'' || z[i] == '"') {
               sqcnt++; // Skip if odd # quotes
            } else if (z[i] == '\n') {
               sqcnt = 0;
            }
            if (z[i] == '<' && z[i+1] == '/' &&
               strncasecmp(&z[i], zEnd, nEnd) == 0) {
               if (zEnd[3] == 'c' && ((sqcnt % 2) == 1)) continue;
               pScr->nScript = i - n;
               pScript = 0;
               n = i + nEnd;
               break;
            }
         }
         if (z[i] == 0) goto incomplete;
         if (pScript) {
            pScr->nScript = i - n;
            n = i;
         } else {
#if 0
        // If there is a script, execute it now and insert any output
        // to the html stream for parsing as html. (ie. client side scripting)

        if (pIsInScript && !pIsInNoScript && !pIsInNoFrames) {

          //for (curch = 0, curline = 1; curch <= curlast; curch++)
          //  if (z[curch] == '\n') curline++;

          // arglist in pElem and text pointers in pScr?
          // Inline scripts can contain unmatched brackets :-)
          //char varind[50];
          //sprintf(varind, "HtmlScrVar%d", p->varind++);
          //char savech = zText[pScr->nStart + pScr->nScript];
          //zText[pScr->nStart + pScr->nScript] = 0;
          //char *scriptBody = StrDup(zText[pScr->nStart]);
          //zText[pScr->nStart + pScr->nScript] = savech;
          AdvanceLayout(p);
          inParse++;
          char *result = ProcessScript((TGHtmlScript *) pElem);  // pElem or pScr??
          inParse--;
          if (result) {
            ol = nAlloc;
            rl = strlen(result);
            nAlloc += rl;
            z = zText = HtmlRealloc(z, ol+rl);
            memmove(z + n + rl, z+n, ol - n);
            memmove(z + n, result, rl);
          }
        }
#endif
         pIsInScript = 0;
         pIsInNoScript = 0;
         pIsInNoFrames = 0;
      }
      //continue;

   } else if (isspace((unsigned char)c)) {

      // White space
      for (i = 0;
           (c = z[n+i]) != 0 && isspace((unsigned char)c) && c != '\n' && c != '\r';
           i++) {}
      if (c == '\r' && z[n+i+1] == '\n') ++i;
#if 0  // this is certainly NOT OK, since it alters pre-formatted text
      if (sawdot == 1) {
        pElem = new TGHtmlTextElement(2);
        strcpy(((TGHtmlTextElement *)pElem)->zText, " ");
        pElem->id = ++idind;
        pElem->offs = n;
        pElem->count = 1; 
        AppendElement(pElem);
      }
#endif
      pElem = new TGHtmlSpaceElement;
      if (pElem == 0) goto incomplete;
      ((TGHtmlSpaceElement *)pElem)->w = 0;
      pElem->offs = n;
      pElem->id = ++idind;
      if (c == '\n' || c == '\r') {
         pElem->flags = HTML_NewLine;
         pElem->count = 1;
         i++;
         inpCol = 0;
      } else {
         int iColStart = inpCol;
         pElem->flags = 0;
         for (j = 0; j < i; j++) {
            inpCol = NextColumn(inpCol, z[n+j]);
         }
         pElem->count = inpCol - iColStart;
      }
      AppendElement(pElem);
      n += i;

      } else if (c != '<' || iPlaintext != 0 || 
         (!isalpha(z[n+1]) && z[n+1] != '/' && z[n+1] != '!' && z[n+1] != '?')) {

      // Ordinary text
      for (i = 1; (c = z[n+i]) != 0 && !isspace((unsigned char)c) && c != '<'; i++) {}
      if (z[n+i-1] == '.' || z[n+i-1] == '!' || z[n+i-1] == '?') sawdot = 2;
      if (c == 0) goto incomplete;
      if (iPlaintext != 0 && z[n] == '<') {
         switch (iPlaintext) {
            case Html_LISTING:
               if (i >= 10 && strncasecmp(&z[n], "</listing>", 10) == 0) {
                  iPlaintext = 0;
                  goto doMarkup;
               }
               break;

            case Html_XMP:
               if (i >= 6 && strncasecmp(&z[n], "</xmp>", 6) == 0) {
                  iPlaintext = 0;
                  goto doMarkup;
               }
               break;

            case Html_TEXTAREA:
               if (i >= 11 && strncasecmp(&z[n], "</textarea>", 11) == 0) {
                  iPlaintext = 0;
                  goto doMarkup;
               }
               break;

            default:
               break;
         }
      }
      pElem = new TGHtmlTextElement(i);
      if (pElem == 0) goto incomplete;
      TGHtmlTextElement *tpElem = (TGHtmlTextElement *) pElem;
      tpElem->id = ++idind;
      tpElem->offs = n;
      strncpy(tpElem->zText, &z[n], i);
      tpElem->zText[i] = 0;
      AppendElement(pElem);
      if (iPlaintext == 0 || iPlaintext == Html_TEXTAREA) {
         HtmlTranslateEscapes(tpElem->zText);
      }
      pElem->count = strlen(tpElem->zText);
      n += i;
      inpCol += i;

   } else if (strncmp(&z[n], "<!--", 4) == 0) {

      // An HTML comment. Just skip it.
      for (i = 4; z[n+i]; i++) {
        if (z[n+i] == '-' && strncmp(&z[n+i], "-->", 3) == 0) break;
      }
      if (z[n+i] == 0) goto incomplete;

      pElem = new TGHtmlTextElement(i);
      if (pElem == 0) goto incomplete;
      TGHtmlTextElement *tpElem = (TGHtmlTextElement *) pElem;
      tpElem->type = Html_COMMENT;
      tpElem->id = ++idind;
      tpElem->offs = n;
      strncpy(tpElem->zText, &z[n+4], i-4);
      tpElem->zText[i-4] = 0;
      tpElem->count = 0;
      AppendElement(pElem);

      pElem = new TGHtmlElement(Html_EndCOMMENT);
      AppToken(pElem, 0, n+4);

      for (j = 0; j < i+3; j++) {
        inpCol = NextColumn(inpCol, z[n+j]);
      }
      n += i + 3;

   } else {

      // Markup.
      //
      // First get the name of the markup
doMarkup:
      argc = 1;
      argv[0] = &z[n+1];
      for (i = 1;
          (c = z[n+i]) != 0 && !isspace((unsigned char)c) && c != '>' && (i < 2 || c != '/');
          i++) {}
      arglen[0] = i - 1;
      if (c == 0) goto incomplete;

      // Now parse up the arguments

      while (isspace((unsigned char)z[n+i])) ++i;
      while ((c = z[n+i]) != 0 && c != '>' && (c != '/' || z[n+i+1] != '>')) {
         if (argc > mxARG - 3) argc = mxARG - 3;
         argv[argc] = &z[n+i];
         j = 0;
         while ((c = z[n+i+j]) != 0 && !isspace((unsigned char)c) && c != '>' &&
                c != '=' && (c != '/' || z[n+i+j+1] != '>')) ++j;
         arglen[argc] = j;
         if (c == 0) goto incomplete;
         i += j;
         while (isspace((unsigned char)c)) {
            i++;
            c = z[n+i];
         }
         if (c == 0) goto incomplete;
         argc++;
         if (c != '=') {
            argv[argc] = "";
            arglen[argc] = 0;
            argc++;
            continue;
         }
         i++;
         c = z[n+i];
         while (isspace((unsigned char)c)) {
            i++;
            c = z[n+i];
         }
         if (c == 0) goto incomplete;
         if (c == '\'' || c == '"') {
            int cQuote = c;
            i++;
            argv[argc] = &z[n+i];
            for (j = 0; (c = z[n+i+j]) != 0 && c != cQuote; j++) {}
            if (c == 0) goto incomplete;
            arglen[argc] = j;
            i += j+1;
         } else {
            argv[argc] = &z[n+i];
            for (j = 0; (c = z[n+i+j]) != 0 && !isspace((unsigned char)c) && c != '>'; j++) {}
            if (c == 0) goto incomplete;
            arglen[argc] = j;
            i += j;
         }
         argc++;
         while (isspace(z[n+i])) ++i;
      }
      if (c == '/') {
         i++;
         c = z[n+i];
         selfClose = 1;
      } else {
         selfClose = 0;
      }
      if (c == 0) goto incomplete;
      for (j = 0; j < i+1; j++) {
         inpCol = NextColumn(inpCol, z[n+j]);
      }
      n += i + 1;

      // Lookup the markup name in the hash table 

      if (!isInit) {
         HtmlHashInit();
         isInit = 1;
      }
      c = argv[0][arglen[0]];
      argv[0][arglen[0]] = 0;
      h = HtmlHash(argv[0]);
      for (pMap = apMap[h]; pMap; pMap = pMap->pCollide) {
        if (strcasecmp(pMap->zName, argv[0]) == 0) break;
      }
      argv[0][arglen[0]] = c;
      if (pMap == 0) continue;  // Ignore unknown markup

makeMarkupEntry:
      // Construct a TGHtmlMarkupElement object for this markup.

      pElem = MakeMarkupEntry(pMap->objType, pMap->type, argc, arglen, argv);
      if (pElem == 0) goto incomplete;

      pElem->id = ++idind;
      pElem->offs = n;

      AddFormInfo(pElem);

      // The new markup has now been constructed in pElem. But before
      // appending it to the list, check to see if there is a special
      // handler for this markup type.

      if (ProcessToken(pElem, pMap->zName, pMap->type)) {
        // delete pElem;

        // Tricky, tricky. The user function might have caused the p->zText
        // pointer to change, so renew our copy of that pointer.

        z = zText;
        if (z == 0) {
          n = 0;
          inpCol = 0;
          goto incomplete;
        }
        continue;
      }

      // No special handler for this markup. Just append it to the
      // list of all tokens.

      AppendElement(pElem);
      switch (pMap->type) {
        case Html_TABLE:
          break;

        case Html_PLAINTEXT:
        case Html_LISTING:
        case Html_XMP:
        case Html_TEXTAREA:
          iPlaintext = pMap->type;
          break;

        case Html_NOFRAMES:
          if (!HasFrames) break;
          pIsInNoFrames = 1;
        case Html_NOSCRIPT:
          break;
          if (!HasScript) break;
          pIsInNoScript = 1;
        case Html_SCRIPT:
          pIsInScript = 1;
        case Html_STYLE:
          pScript = (TGHtmlScript *) pElem;
          break;

        case Html_LI:
          if (!AddEndTags) break;
          if (inLi) {
            TGHtmlElement *e = new TGHtmlMarkupElement(Html_EndLI, 1, 0, 0);
            AppToken(e, pElem, n);
          } else {
            inLi = 1;
          }
          break;

        case Html_EndLI:
          inLi=0;
          break;

        case Html_EndOL:
        case Html_EndUL:
          if (!AddEndTags) break;
          if (inLi) {
            TGHtmlElement *e = new TGHtmlMarkupElement(Html_EndLI, 1, 0, 0);
            AppToken(e, pElem, n);
          } else {
            inLi = 0;
          }
          break;

        default:
          break;
      }

      // If this is self-closing markup (ex: <br/> or <img/>) then
      // synthesize a closing token.

      if (selfClose && argv[0][0] != '/' &&
          strcmp(&pMap[1].zName[1], pMap->zName) == 0) {
        selfClose = 0;
        pMap++;
        argc = 1;
        goto makeMarkupEntry;
      }
    }
  }

incomplete:
  iCol = inpCol;
  ////pScript = 0;

  return n;
}

/************************** End HTML Tokenizer Code ***************************/

//______________________________________________________________________________
TGHtmlMarkupElement *TGHtml::MakeMarkupEntry(int objType, int type, int argc,
                                            int arglen[], char *argv[])
{
   //

   TGHtmlMarkupElement *e;

   switch (objType) {
    case O_HtmlCell:
      e = new TGHtmlCell(type, argc, arglen, argv);
      break;

    case O_HtmlTable:
      e = new TGHtmlTable(type, argc, arglen, argv);
      break;

    case O_HtmlRef:
      e = new TGHtmlRef(type, argc, arglen, argv);
      break;

    case O_HtmlLi:
      e = new TGHtmlLi(type, argc, arglen, argv);
      break;

    case O_HtmlListStart:
      e = new TGHtmlListStart(type, argc, arglen, argv);
      break;

    case O_HtmlImageMarkup:
      e = new TGHtmlImageMarkup(type, argc, arglen, argv);
      break;

    case O_HtmlInput:
      e = new TGHtmlInput(type, argc, arglen, argv);
      break;

    case O_HtmlForm:
      e = new TGHtmlForm(type, argc, arglen, argv);
      break;

    case O_HtmlHr:
      e = new TGHtmlHr(type, argc, arglen, argv);
      break;

    case O_HtmlAnchor:
      e = new TGHtmlAnchor(type, argc, arglen, argv);
      break;

    case O_HtmlScript:
      e = new TGHtmlScript(type, argc, arglen, argv);
      break;

    case O_HtmlMapArea:
      e = new TGHtmlMapArea(type, argc, arglen, argv);
      break;

    default:
      e = new TGHtmlMarkupElement(type, argc, arglen, argv);
      break;
  }

  return e;
}

//______________________________________________________________________________
void TGHtml::TokenizerAppend(const char *text)
{
   // Append text to the tokenizer engine.

   int len = strlen(text);

   if (nText == 0) {
      nAlloc = len + 100;
      zText = new char [nAlloc];
   } else if (nText + len >= nAlloc) {
      nAlloc += len + 100;
      char *tmp = new char[nAlloc];
      strcpy(tmp, zText);
      delete[] zText;
      zText = tmp;
   }

  if (zText == 0) {
    nText = 0;
    UNTESTED;
    return;
  }

  strcpy(&zText[nText], text);
  nText += len;
  nComplete = Tokenize();
}

//______________________________________________________________________________
TGHtmlElement *TGHtml::InsertToken(TGHtmlElement *pToken,
                                  char *zType, char *zArgs, int offs)
{
   // This routine takes a text representation of a token, converts it into an
   // TGHtmlElement object and inserts it immediately prior to pToken. If pToken
   // is 0, then the newly created TGHtmlElement is appended.
   //
   // This routine does nothing to resize, restyle, relayout or redisplay
   // the HTML. That is the calling routines responsibility.
   //
   // Return the new TGHtmlElement object if successful. Return zero if
   // zType is not a known markup name.
   //
   //  pToken  - Insert before this. Append if pToken == 0
   //  zType   - Type of markup. Ex: "/a" or "table"
   //  zArgs   - List of arguments
   //  offs    - Calculate offset, and insert changed text into zText!

   SHtmlTokenMap *pMap;     // For searching the markup name hash table
   int h;                   // The hash on zType
   TGHtmlElement *pElem;     // The new element
   //int nByte;               // How many bytes to allocate
   //int i;                   // Loop counter

   if (!isInit) {
      HtmlHashInit();
      isInit = 1;
   }

   if (strcmp(zType, "Text") == 0) {
      pElem = new TGHtmlTextElement(zArgs ? strlen(zArgs) : 0);
      if (pElem == 0) return 0;
      if (zArgs) {
         strcpy (((TGHtmlTextElement *)pElem)->zText, zArgs);
         pElem->count = strlen(zArgs);
      }
   } else if (!strcmp(zType, "Space")) {
      pElem = new TGHtmlSpaceElement();
      if (pElem == 0) return 0;
   } else {
      h = HtmlHash(zType);
      for (pMap = apMap[h]; pMap; pMap = pMap->pCollide) {
         if (strcasecmp(pMap->zName, zType) == 0) break;
      }
      if (pMap == 0) return 0;
      if (zArgs == 0 || *zArgs == 0) {
         // Special case of no arguments. This is a lot easier...
         // well... now its the same thing!
         pElem = MakeMarkupEntry(pMap->objType, pMap->type, 1, 0, 0);
         if (pElem == 0) return 0;
      } else {
         // The general case. There are arguments that need to be parsed
         // up. This is slower, but we gotta do it.
         //int  argc;
         //char **argv;
         //char *zBuf;

#if 0
      if (!SplitList(zArgs, &argc, &argv)) return 0;

      // shall we insert a dummy argv[0]?

      pElem = MakeMarkupEntry(pMap->objType, pMap->type, argc/*+1??*/, 0, argv);
      if (pElem == 0) return 1;

      while (--argc >= 0) if (argv[argc]) delete[] argv[argc];
      delete[] argv;
#else
         return 0;
#endif
      }
   }

   pElem->id = ++idind;

   AppToken(pElem, pToken, offs);

   return pElem;
}

//______________________________________________________________________________
int TGHtml::TextInsertCmd(int /*argc*/, char ** /*argv*/)
{
   // Insert text into text token, or break token into two text tokens.
   // Also, handle backspace char by deleting text.
   // Should also handle newline char by splitting text.

#if 0
  TGHtmlElement *p, *pElem;
  int i, l, n = 0;
  int idx = 0;
  int ptyp = Html_Unknown;
  int istxt = 0;
  char *cp = 0, c, *cp2;

  if (GetIndex(argv[3], &p, &i) != 0) {
    // sprintf(tmp, "malformed index: \"%s\"", argv[3]);
    return 0;
  }
  if (p) {
    ptyp = p->type;
    if ((istxt = (ptyp == Html_Text))) {
      l = p->count;
      cp = ((TGHtmlTextElement *)p)->zText;
    }
  }
  if (argv[2][0] == 'b') {  // Break text token into two.
    if (!istxt) return 1;
    if (i == 0 || i == l) return 1;
    pElem = InsertToken(p->pNext, "Text", cp + i, -1);
    cp[i] = 0;
    p->count = i;
    return 1;
  }
  c = argv[4][0];
  if (!c) return 1;
  if (c == '\b') {
    if ((!istxt) || (!l) || (!i)) {
      if (!p) return 1;
      if (p->type == Html_BR)
        RemoveElements(p, p);
      return 1;
    }
    if (p && l == 1) {
      RemoveElements(p, p);
      return 1;
    }
    if (i == l)
      cp[p->count] = 0;
    else
      memcpy(cp+i-1, cp+i, l-i+1);

    cp[--p->count] = 0;
    if (ins.i-- <= 0) ins.i = 0;
    ins.p = p;
    return 1;
  }
  if (c == '\n' || c == '\r') {
  }
  if (istxt) {
    char *cp;
    int t, j, alen = strlen(argv[4]);
    n = alen + l;

    TGHtmlTextElement *text = (TGHtmlTextElement *) p;

    if (text->zText == (char*) ((&text->zText)+1)) {
      cp = new char[n+1];
      strcpy(cp, text->zText);
    } else {
      cp = new char[n+1];
      strcpy(cp, text->zText);
    }
    cp2 = new char[alen+1];
    memcpy(cp2, argv[4], alen+1);
    HtmlTranslateEscapes(cp2);
    alen = strlen(cp2);
    memmove(cp+alen+i, cp+i, l-i+1);
    for (j = 0; j < alen; j++) cp[i+j] = cp2[j];
    delete[] cp2;
    delete[] text->zText;
    text->zText = cp;
    p->count = strlen(cp);
    ins.p = p;
    ins.i = i+alen;
  } else {
    p = InsertToken(p ? p->pNext : 0, "Text", argv[4], -1);
    AddStyle(p);
    i = 0;
    ins.p = p;
    ins.i = 1;
  }
  if (p) {
    idx = p->base.id;
    AddStrOffset(p, argv[4], i);
  }
#endif
  return 1;
}

//______________________________________________________________________________
SHtmlTokenMap *TGHtml::NameToPmap(char *zType)
{
   //

   SHtmlTokenMap *pMap;     // For searching the markup name hash table
   int h;                   // The hash on zType

   if (!isInit) {
      HtmlHashInit();
      isInit = 1;
   }
   h = HtmlHash(zType);
   for (pMap = apMap[h]; pMap; pMap = pMap->pCollide) {
      if (strcasecmp(pMap->zName, zType) == 0) break;
   }

   return pMap;
}

//______________________________________________________________________________
int TGHtml::NameToType(char *zType)
{
   // Convert a markup name into a type integer

   SHtmlTokenMap *pMap = NameToPmap(zType);
   return pMap ? pMap->type : (int)Html_Unknown;
}

//______________________________________________________________________________
const char *TGHtml::TypeToName(int type)
{
   // Convert a type into a symbolic name

   if (type >= Html_A && type <= Html_EndXMP) {
      SHtmlTokenMap *pMap = apMap[type - Html_A];
      return pMap->zName;
   } else {
      return "???";
   }
}

//______________________________________________________________________________
char *TGHtml::DumpToken(TGHtmlElement *p)
{
   // For debugging purposes, print information about a token

//#ifdef DEBUG
   static char zBuf[200];
   int j;
   char *zName;

   if (p == 0) return "NULL";
   switch (p->type) {
      case Html_Text:
         sprintf(zBuf, "text: \"%.*s\"", p->count, ((TGHtmlTextElement *)p)->zText);
         break;

      case Html_Space:
         if (p->flags & HTML_NewLine) {
            sprintf(zBuf, "space: \"\\n\"");
         } else {
            sprintf(zBuf, "space: \" \"");
         }
         break;

      case Html_Block: {
         TGHtmlBlock *block = (TGHtmlBlock *) p;
         if (block->n > 0) {
            int n = block->n;
            if (n > 150) n = 150;
               sprintf(zBuf, "<Block z=\"%.*s\">", n, block->z);
            } else {
               sprintf(zBuf, "<Block>");
            }
            break;
      }

      default:
         if (p->type >= HtmlMarkupMap[0].type 
             && p->type <= HtmlMarkupMap[HTML_MARKUP_COUNT-1].type) {
            zName = HtmlMarkupMap[p->type - HtmlMarkupMap[0].type].zName;
         } else {
            zName = "Unknown";
         }
         sprintf(zBuf, "markup (%d) <%s", p->type, zName);
         for (j = 1 ; j < p->count; j += 2) {
            sprintf(&zBuf[strlen(zBuf)], " %s=\"%s\"",
                    ((TGHtmlMarkupElement *)p)->argv[j-1],
                    ((TGHtmlMarkupElement *)p)->argv[j]);
         }
         strcat(zBuf, ">");
         break;
   }
   return zBuf;
//#else
//  return 0;
//#endif
}

//______________________________________________________________________________
void TGHtml::AppendArglist(TGString *str, TGHtmlMarkupElement *pElem)
{
   // Append all the arguments of the given markup to the given TGString.
   //
   // Example:  If the markup is <IMG SRC=image.gif ALT="hello!">
   // then the following text is appended to the TGString:
   //
   //       "src image.gif alt hello!"
   //
   // Notice how all attribute names are converted to lower case.
   // This conversion happens in the parser.

   int i;

   for (i = 0; i + 1 < pElem->count; i += 2) {
      str->Append(pElem->argv[i]);
      str->Append("=");
      str->Append(pElem->argv[i+1]);
      str->Append(" ");
   }
}

//______________________________________________________________________________
char *TGHtml::GetTokenName(TGHtmlElement *p)
{
   //

   static char zBuf[200];
   //int j;
   char *zName;

   zBuf[0] = 0;
   if (p == 0) return "NULL";
   switch (p->type) {
      case Html_Text:
      case Html_Space:
         break;

      case Html_Block:
         break;

      default:
         if (p->type >= HtmlMarkupMap[0].type &&
             p->type <= HtmlMarkupMap[HTML_MARKUP_COUNT-1].type) {
            zName = HtmlMarkupMap[p->type - HtmlMarkupMap[0].type].zName;
         } else {
            zName = "Unknown";
         }
         strcpy(zBuf, zName);
         break;
   }

   return zBuf;
}

//______________________________________________________________________________
SHtmlTokenMap* TGHtml::GetMarkupMap(int n)
{
   //

   return HtmlMarkupMap+n;
}

//______________________________________________________________________________
TGString *TGHtml::ListTokens(TGHtmlElement *p, TGHtmlElement *pEnd)
{
   // Return all tokens between the two elements as a string list.

   TGString *str;
   int i;
   char *zName;
   char zLine[100];

   str = new TGString("");
   while (p && p != pEnd) {
      switch (p->type) {
         case Html_Block:
            break;

         case Html_Text:
            str->Append("{ Text \"");
            str->Append(((TGHtmlTextElement *)p)->zText);
            str->Append("\" } ");
            break;

         case Html_Space:
            sprintf(zLine, "Space %d %d ",
                    p->count, (p->flags & HTML_NewLine) != 0);
            str->Append(zLine);
            break;

         case Html_Unknown:
            str->Append("Unknown ");
            break;

         default:
            str->Append("{ Markup ");
            if (p->type >= HtmlMarkupMap[0].type &&
                p->type <= HtmlMarkupMap[HTML_MARKUP_COUNT-1].type) {
               zName = HtmlMarkupMap[p->type - HtmlMarkupMap[0].type].zName;
            } else {
               zName = "Unknown";
            }
            str->Append(zName);
            str->Append(" ");
            for (i = 0; i < p->count; ++i) {
               str->Append(((TGHtmlMarkupElement *)p)->argv[i]);
               str->Append(" ");
            }
            str->Append("} ");
            break;
      }
      p = p->pNext;
   }

   return str;
}

//______________________________________________________________________________
void TGHtml::PrintList(TGHtmlElement *first, TGHtmlElement *last)
{
   // Print a list of tokens

   TGHtmlElement *p;

   for (p = first; p != last; p = p->pNext) {
      if (p->type == Html_Block) {
         TGHtmlBlock *block = (TGHtmlBlock *) p;
         char *z = block->z;
         int n = block->n;
         if (n == 0 || z == 0) {
            n = 1;
            z = "";
         }
         printf("Block flags=%02x cnt=%d x=%d..%d y=%d..%d z=\"%.*s\"\n",
                 p->flags, p->count, block->left, block->right,
         block->top, block->bottom, n, z);
      } else {
         printf("Token font=%2d color=%2d align=%d flags=0x%04x name=%s\n",
                p->style.font, p->style.color,
                p->style.align, p->style.flags, DumpToken(p));
      }
   }
}
