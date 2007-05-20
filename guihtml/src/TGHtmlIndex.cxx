// $Id: TGHtmlIndex.cxx,v 1.1 2007/05/04 17:07:01 brun Exp $
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

// Routines that deal with indexes

#include <ctype.h>
#include <string.h>

#include "TGHtml.h"


//______________________________________________________________________________
TGHtmlElement *TGHtml::TokenByIndex(int N, int /*flag*/)
{
   // Return a pointer to the Nth TGHtmlElement in the list. If there
   // is no Nth element, return 0 if flag==0 and return either the first
   // or last element (whichever is closest) if flag!=0

   TGHtmlElement *p;
   int n;

   if (N == 0) return pFirst;

   if (N > nToken / 2) {
      // Start at the end and work back toward the beginning
      for (p = pLast, n = nToken; p; p = p->pPrev) {
         if (p->type != Html_Block) {
            if (p->id == N) break;
            --n;
         }
      }
   } else {
      // Start at the beginning and work forward
      for (p = pFirst; p; p = p->pNext) {
         if (p->type != Html_Block) {
            --N;
            if (N == p->id) break;
         }
      }
   }

   return p;
}

//______________________________________________________________________________
int TGHtml::TokenNumber(TGHtmlElement *p)
{
   // Return the token number for the given TGHtmlElement

   //int n = 0;

   if (!p) return -1;
   return p->id;

///  while (p) {
///    if (p->type != Html_Block) ++n;
///    p = p->pPrev;
///  }
///
///  return n;
}

//______________________________________________________________________________
void TGHtml::maxIndex(TGHtmlElement *p, int *pIndex, int isLast)
{
   // Find the maximum index for the given token

   if (p == 0) {
      *pIndex = 0;
   } else {
      switch (p->type) {
         case Html_Text:
            *pIndex = p->count - isLast;
            break;

         case Html_Space:
            if (p->style.flags & STY_Preformatted) {
               *pIndex = p->count - isLast;
            } else {
               *pIndex = 0;
            }
            break;

         default:
            *pIndex = 0;
            break;
      }
   }
}

//______________________________________________________________________________
void TGHtml::FindIndexInBlock(TGHtmlBlock *pBlock, int x,
                              TGHtmlElement **ppToken, int *pIndex)
{
   // Given a Block and an x coordinate, find the Index of the character
   // that is closest to the given x coordinate.
   //
   // The x-coordinate might specify a point to the left of the block,
   // in which case the procedure returns the first token and a character
   // index of 0.  Or the x-coordinate might specify a point to the right
   // of the block, in which case the last token is returned with an index
   // equal to its last character.

   TGHtmlElement *p;
   TGFont *font;
   int len;
   int n;

   p = pBlock->pNext;
   font = GetFont(p->style.font);
   if (x <= pBlock->left) {
      *ppToken = p;
      *pIndex = 0;
      return;
   } else if (x >= pBlock->right) {
      *ppToken = p;
      *pIndex = 0;
      while (p && p->type != Html_Block) {
         *ppToken = p;
         p = p->pNext;
      }
      p = *ppToken;
      if (p && p->type == Html_Text) {
         *pIndex = p->count - 1;
      }
      return;
   }
   if (pBlock->n == 0) {
      *ppToken = p;
      *pIndex = 0;
   }
   n = font->MeasureChars(pBlock->z, pBlock->n, x - pBlock->left, 0, &len);
   *pIndex = 0;
   *ppToken = 0;
   while (p && n >= 0) {
      switch (p->type) {
         case Html_Text:
            if (n < p->count) {
               *pIndex = n;
            } else {
               *pIndex = p->count - 1;
            }
            *ppToken = p;
            n -= p->count;
            break;

         case Html_Space:
            if (p->style.flags & STY_Preformatted) {
               if (n < p->count) {
                  *pIndex = n;
               } else {
                  *pIndex = p->count - 1;
               }
               *ppToken = p;
               n -= p->count;
            } else {
               *pIndex = 0;
               *ppToken = p;
               --n;
            }
            break;

         default:
            break;
      }
      if (p) p = p->pNext;
   }
}

//______________________________________________________________________________
void TGHtml::IndexToBlockIndex(SHtmlIndex sIndex,
                               TGHtmlBlock **ppBlock, int *piIndex)
{
   // Convert an Element-based index into a Block-based index.
   //
   // In other words, given a pointer to an element and an index
   // of a particular character within that element, compute a
   // pointer to the TGHtmlBlock used to display that character and
   // the index in the TGHtmlBlock of the character.

   int n = sIndex.i;
   TGHtmlElement *p;

   if (sIndex.p == 0) {
      *ppBlock = 0;
      *piIndex = 0;
      return;
   }
   p = sIndex.p->pPrev;
   while (p && p->type != Html_Block) {
      switch (p->type) {
         case Html_Text:
            n += p->count;
            break;
         case Html_Space:
            if (p->style.flags & STY_Preformatted) {
               n += p->count;
            } else {
               n++;
            }
            break;
         default:
            break;
      }
      p = p->pPrev;
   }
   if (p) {
      *ppBlock = (TGHtmlBlock *) p;
      *piIndex = n;
      return;
   }
   for (p = sIndex.p; p && p->type != Html_Block; p = p->pNext) {}
   *ppBlock = (TGHtmlBlock *) p;
   *piIndex = 0;
}

//______________________________________________________________________________
int TGHtml::IndexMod(TGHtmlElement **pp, int *ip, char *cp)
{
   // Modify an index for both pointer and char +/-/=N

   char nbuf[50];
   int i, x, cnt, ccnt[2], cflag[2];

   if (pp == 0 || !*pp) return -1;
   ccnt[0] = ccnt[1] = cflag[0] = cflag[1] = 0;
   x = 0;
   while (*cp && x < 2) {
      cnt = 0;
      i = 1;
      while (i < 45 && isdigit(cp[i])) {
         nbuf[i-1] = cp[i];
         i++;
      }
      if (i > 1) {
         nbuf[i-1] = 0;
         cnt = atoi(nbuf);
         if (cnt < 0) return -1;
      }
      switch (*cp) {
         case '+': if (i == 1) ccnt[x] = 1; else ccnt[x] = cnt; break;
         case '-': if (i == 1) ccnt[x] = -1; else ccnt[x] = -cnt; break;
         case '=': ccnt[x] = 0; cflag[x] = 1; break;
         default: return -1;
      }
      cp += i;
      ++x;
   }
   if (ccnt[0] > 0) {
      for (i = 0; i < ccnt[0] && (*pp)->pNext; ++i) {
         *pp = (*pp)->pNext;
         while ((*pp)->type == Html_Block && (*pp)->pNext) {
            *pp = (*pp)->pNext;
         }
      }
   } else if (ccnt[0] < 0) {
      for (i = 0; ccnt[0] < i && (*pp)->pPrev; --i) {
         //printf("i=%d, cnt=%d\n", i, ccnt[0]);
         *pp = (*pp)->pPrev;
         while ((*pp)->type == Html_Block && (*pp)->pPrev) {
            *pp = (*pp)->pPrev;
         }
      }
   }
   if (ccnt[1] > 0) {
      for (i = 0; i < ccnt[1]; ++i) (*ip)++;
   } else if (ccnt[1] < 0) {
      for (i = 0; i > ccnt[1]; --i) (*ip)--;
   }
   return 0;
}

//______________________________________________________________________________
int TGHtml::DecodeBaseIndex(const char *baseIx,
                            TGHtmlElement **ppToken, int *pIndex)
{
   // Given a base index name (without any modifiers) return a pointer
   // to the token described, and the character within that token.
   //
   // Valid input forms include:
   //
   //       N.M          Token number N (with numbering starting at 1) and
   //                    character number M (with numbering starting at 0).
   //
   //       M.X          Like above, but token is markup and X is an attribute.
   //
   //       begin        The start of all text
   //
   //       end          The end of all text
   //
   //       N.last       Last character of token number N.
   //
   //       N.end        One past last character of token number N.
   //
   //       sel.first    First character of the selection.
   //
   //       sel.last     Last character of the selection.
   //
   //       sel.end      On past last character of the selection.
   //
   //       insert       The character holding the insertion cursor.
   //
   //       @X,Y         The character a location X,Y of the clipping window.
   //
   //       &DOM         The DOM Address of a token.
   //
   // Zero is returned if we are successful and non-zero if there is
   // any kind of error.
   //
   // If the given token doesn't exist (for example if there are only 10
   // tokens and 11.5 is requested) then *ppToken is left pointing to NULL.
   // But the function still returns 0 for success.

   int i, n, x, y;
   TGHtmlElement *p;
   TGHtmlBlock *pBlock;
   TGHtmlBlock *pNearby;
   int dist = 1000000;
   int rc = 0;
   char buf[200], *base = buf, *suffix, *ep;

   strncpy(buf, baseIx, sizeof(buf));
   buf[sizeof(buf) - 1] = 0;

   while (isspace((unsigned char)*base)) base++;
   ep = base;
   while (*ep && !isspace((unsigned char)*ep)) ep++;
   *ep = 0;

   if ((suffix = strchr(base, ':'))) *suffix = 0;

   switch (*base) {
      case '1': case '2': case '3': case '4': case '5':
      case '6': case '7': case '8': case '9': case '0':
         n = sscanf(base, "%d.%d", &x, &y);
         if (n > 0) {
            p = *ppToken = TokenByIndex(x, 0);
         }
         if (n == 2) {
            *pIndex = y;
         } else {
            for (i = 1; isdigit(base[i]); ++i) {}
            if (base[i] == 0) {
               *pIndex = 0;
            } else if (strcmp(&base[i], ".last") == 0) {
               maxIndex(p, pIndex, 1);
            } else if (strcmp(&base[i], ".end") == 0) {
               maxIndex(p, pIndex, 0);
               (*pIndex)++;
            } else {
               if (n == 1 && p && p->IsMarkup() && base[i] == '.' &&
                   p->MarkupArg(zBase + i + 1, 0)) {
                  *pIndex = 0;
               } else {
                  rc = 1;
               }
            }
         }
         break;

      case 'b':
         if (strcmp(base, "begin") == 0) {
            p = *ppToken = pFirst;
            *pIndex = 0;
         } else {
            rc = 1;
         }
         break;

      case 'e':
         if (strcmp(base, "end") == 0) {
            p = *ppToken = pLast;
            maxIndex(p, pIndex, 0);
         } else {
            rc = 1;
         }
         break;

      case 'l':
         if (strcmp(base, "last") == 0) {
            p = *ppToken = pLast;
            maxIndex(p, pIndex, 1);
         } else {
            rc = 1;
         }
         break;

      case 's':
         if (strcmp(base, "sel.first") == 0) {
            *ppToken = selBegin.p;
            *pIndex = selBegin.i;
         } else if (strcmp(base, "sel.last") == 0) {
            *ppToken = selEnd.p;
            *pIndex = selEnd.i;
         } else if (strcmp(base, "sel.end") == 0) {
            *ppToken = selEnd.p;
            *pIndex = selEnd.i + 1;
         } else {
            rc = 1;
         }
         break;

      case 'i':
         if (strcmp(baseIx, "insert") == 0) {
            *ppToken = ins.p;
            *pIndex = ins.i;
         } else {
            rc = 1;
         }
         break;

#if 0
    case '&':
      *pIndex = 0;
      if (DomIdLookup("id", base + 1, ppToken)) rc = 1;
      break;
#endif

      case '@':
         n = sscanf(base, "@%d,%d", &x, &y);
         if (n != 2) {
            rc = 1;
            break;
         }
         x += fVisible.fX;
         y += fVisible.fY;
         pNearby = 0;
         *ppToken = pLast;
         *pIndex = 0;
         for (pBlock = firstBlock; pBlock; pBlock = pBlock->bNext) {
            int dotest;
            if (pBlock->n == 0) {
               switch (pBlock->pNext->type) {
                  case Html_LI:
                  case Html_IMG:
                  case Html_INPUT:
                  case Html_TEXTAREA:
                  case Html_SELECT:
                     dotest = 1;
                     break;
                  default:
                     dotest = 0;
                     break;
               }
            } else {
               dotest = 1;
            }
            if (dotest) {
               if (pBlock->top <= y && pBlock->bottom >= y) {
                  if (pBlock->left > x) {
                     if (pBlock->left - x < dist) {
                        dist = pBlock->left - x;
                        pNearby = pBlock;
                     }
                  } else if (pBlock->right < x) {
                     if (x - pBlock->right < dist) {
                        dist = x - pBlock->right;
                        pNearby = pBlock;
                     }
                  } else {
                     FindIndexInBlock(pBlock, x, ppToken, pIndex);
                     break;
                  }
               } else {
                  int distY;
                  int distX;

                  if (pBlock->bottom < y) {
                     distY = y - pBlock->bottom;
                  } else {
                     distY = pBlock->top - y;
                  }
                  if (pBlock->left > x) {
                     distX = pBlock->left - x;
                  } else if (pBlock->right < x) {
                     distX = x - pBlock->right;
                  } else {
                     distX = 0;
                  }
                  if (distX + 4*distY < dist) {
                     dist = distX + 4*distY;
                     pNearby = pBlock;
                  }
               }
            }
         }
         if (pBlock == 0) {
            if (pNearby) {
               FindIndexInBlock(pNearby, x, ppToken, pIndex);
            }
         }
         break;

      default:
         rc = 1;
         break;
   }
   if (suffix) IndexMod(ppToken, pIndex, suffix + 1);

   return rc;
}

//______________________________________________________________________________
int TGHtml::GetIndex(const char *zIndex,
                     TGHtmlElement **ppToken, int *pIndex)
{
   // This routine decodes a complete index specification. A complete
   // index consists of the base specification followed by modifiers.

   return DecodeBaseIndex(zIndex, ppToken, pIndex);
}
