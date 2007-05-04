// $Id$
// Author:  Valeriy Onuchin   03/05/2007

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

// Routines for doing layout of HTML tables

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "TGHtml.h"


// Default values for various table style parameters

#define DFLT_BORDER             0
#define DFLT_CELLSPACING_3D     5
#define DFLT_CELLSPACING_FLAT   0
#define DFLT_CELLPADDING        2
#define DFLT_HSPACE             0
#define DFLT_VSPACE             0

// Set parameter A to the maximum of A and B.
#define SETMAX(A,B)  if ((A) < (B)) { (A) = (B); }
#define MAX(A,B)     ((A) < (B) ? (B) : (A))


//______________________________________________________________________________
int TGHtml::CellSpacing(TGHtmlElement *pTable)
{
   // Return the appropriate cell spacing for the given table.

   char *z;
   int relief;
   int cellSpacing;

   z = pTable->MarkupArg("cellspacing", 0);
   if (z == 0) {
      relief = tableRelief;
      if (relief == HTML_RELIEF_RAISED || relief == HTML_RELIEF_SUNKEN) {
         cellSpacing = DFLT_CELLSPACING_3D;
      } else {
         cellSpacing = DFLT_CELLSPACING_FLAT;
      }
   } else {
      cellSpacing = atoi(z);
   }

   return cellSpacing;
}

//______________________________________________________________________________
void TGHtml::StringHW(const char *str, int *h, int *w)
{
   // Return the height and width of string.

   const char *cp = str;
   int nw = 0, nh = 1, mw = 0;
   *h = 0; *w =0;

   if (!cp) return;

   while (*cp) {
      if (*cp != '\n') {
         nw++;
      } else {
         if (nw > mw) mw = nw;
         nw = 0;
         nh++;
      }
      cp++;
   }
   if (nw > mw) mw = nw;
   *w = mw;
   *h = nh;
}

//______________________________________________________________________________
TGString *TGHtml::TableText(TGHtmlTable *pTable, int flags)
{
   // Return text and images from a table as lists.
   // The first list is a list of rows (which is a list of cells).
   // An optional second list is a list of images: row col charoffset tokenid.
   // Note: weve added the option to store data/attrs in array var directly.
   //
   // flags - include images

   int j, h, w, 
      Nest = 0,
 //     intext = 0,
      rows = 0,
      cols = 0,
      numcols = 0,
      maxh = 1;
   int cspans = 0,
      rspanstart = 0,
      images = flags & 1,
      attrs = flags & 2;
   unsigned short maxw[HTML_MAX_COLUMNS];
   short rspans[HTML_MAX_COLUMNS];
   char buf[100];
   const char *cp;
   TGHtmlElement *p, *pEnd;
   TGString istr("");     // Information string
   TGString substr("");   // Temp to collect current cell string.
   TGString imgstr("");   // Image information
   TGString attrstr("");  // Attribue information

   TGString *str = new TGString("");  // The result

   if (pTable->type != Html_TABLE) return str;
   if (!(pEnd = pTable->pEnd)) {
      delete str;
      return 0;
   }

   str->Append("{ ");  // start sublist
   if (attrs) {
      attrstr.Append("{ ");  // start sublist
      AppendArglist(&attrstr, pTable);
      attrstr.Append("} ");  // end sublist
   }
   for (j = 0; j < HTML_MAX_COLUMNS; j++) {
      maxw[j] = 0;
      rspans[j] = 0;
   }
   Nest = 1;
   istr.Append("{ ");
   p = pTable;
   while (p && (p = p->pNext)) {
      if (attrs) {
         switch (p->type) {
            case Html_EndTR:
               break;

            case Html_TR:
               break;
         }
      }

      switch (p->type) {
         case Html_TABLE:
            if (!(p = FindEndNest(p, Html_EndTABLE, 0))) {
               delete str;
               return 0;
            }
            break;

         case Html_EndTABLE:
            p = 0;
            break;

         case Html_TR:
            if (cols > numcols) numcols = cols;
            maxh = 1;
            cols = 0;
            rows++;
            Nest++;
            str->Append("{ ");
            if (attrs) {
               attrstr.Append("{ { ");
               AppendArglist(&attrstr, (TGHtmlMarkupElement *) p);
               attrstr.Append("} ");
            }
            break;

         case Html_EndTR:
            sprintf(buf, "%d ", maxh);
            istr.Append(buf);
            if (attrs) {
               attrstr.Append("} ");
            }
            while (Nest > 1) {
               Nest--;
               str->Append("} ");
            }
            break;

         case Html_TD:
         case Html_TH:
            if ((!(cp = p->MarkupArg("colspan", 0))) || (cspans = atoi(cp)) <= 0) {
               cspans = 1;
            }
            if ((cp = p->MarkupArg("rowspan", 0)) && (j = atoi(cp)) > 0 &&
                cols < HTML_MAX_COLUMNS) {
               rspans[cols] = j;
               rspanstart = 1;
            } else {
               rspanstart = 0;
            }
            if (attrs) {
               j = 0;
               while ((cspans - j) > 0) {
                  attrstr.Append("{ ");
                  if (!j) AppendArglist(&attrstr, (TGHtmlMarkupElement *) p);
                  attrstr.Append("} ");
                  j++;
               }
            }
            cols++;
            substr = "";
            break;

         case Html_EndTD:
         case Html_EndTH:
            if (!rspanstart) {
               while (cols <= HTML_MAX_COLUMNS && rspans[cols-1]-- > 1) {
                  str->Append(" ");  // (""); ??
                  cols++;
               }
            }
            cp = substr.GetString();
          
            j = 0;
            while ((cspans - j) > 0) {
               str->Append(cp);
               str->Append(" ");
               if (!j) {
                  StringHW(cp, &h, &w);
                  if (h > maxh) maxh = h;
                  if (cols > 0 && cols <= HTML_MAX_COLUMNS) {
                     if (w > maxw[cols-1]) {
                        maxw[cols-1] = w;
                     }
                  }
               }
               j++;
               cp = "";
            }
            cspans = 0;
            break;

         case Html_Text:
            substr.Append(((TGHtmlTextElement *)p)->zText, -1);
            break;

         case Html_Space:
            for (j = 0; j < p->count; j++) {
               substr.Append(" ");
            }
//        if ((p->flags & HTML_NewLine) != 0)
//          substr.Append("\n");
            break;

         case Html_BR:
            substr.Append("\n");  // ("\\n"); ??
            break;

         case Html_CAPTION:  // Should do something with Caption?
            if (!(pEnd = FindEndNest(p, Html_EndCAPTION, 0))) {
               p = pEnd;
            }
            break;

         case Html_IMG:  // Images return: row col charoffset tokenid
            if (!images) break;
            sprintf(buf, "%d %d %d %d ", rows-1, cols-1, 
                    substr.GetLength(), p->id);
            imgstr.Append(buf);
            break;
      }
   }

   while (Nest--) str->Append("} ");
   istr.Append("} { ");
   for (j = 0; j < numcols && j < HTML_MAX_COLUMNS; j++) {
      sprintf(buf, "%d ", maxw[j]);
      istr.Append(buf);
   }
   istr.Append("} ");

   str->Append(istr.Data());
   str->Append(" ");
   if (attrs) {
      str->Append("{ ");
      str->Append(attrstr.Data());
      str->Append("} ");
   }
   if (images) {
      str->Append(imgstr.Data());
   }

   return str;
}

//______________________________________________________________________________
TGHtmlElement *TGHtml::FindEndNest(TGHtmlElement *sp, int en,
                                  TGHtmlElement *lp)
{
   // Find End tag en, but ignore intervening begin/end tag pairs.
   //
   // sp -- Pointer to start from
   // en -- End tag to search for
   // lp -- Last pointer to try

   TGHtmlElement *p;
   int lvl, n;

   p = sp->pNext;
   lvl = 0;
   n = sp->type;

   while (p) {
      if (p == lp) return 0;
      if (n == Html_LI) {
         if (p->type == Html_LI || p->type == Html_EndUL ||
             p->type == Html_EndOL) {
            if (p->pPrev) return p->pPrev;
            return p;
         }
      } else if (p->type == n) {
         if (n == Html_OPTION) {
            if (p->pPrev) return p->pPrev;
            return p;
         }
         lvl++;
      } else if (p->type == en) {
         if (!lvl--) return p;
      }
      switch (p->type) {
         case Html_TABLE: p = ((TGHtmlTable *)p)->pEnd; break; // optimization
         case Html_FORM:  p = ((TGHtmlForm *)p)->pEnd;  break;
         default: p = p->pNext;
      }
   }

   return 0;
}

//______________________________________________________________________________
TGHtmlElement *TGHtml::TableDimensions(TGHtmlTable *pStart, int lineWidth)
{
   // pStart points to a <table>.  Compute the number of columns, the
   // minimum and maximum size for each column and the overall minimum
   // and maximum size for this table and store these value in the
   // pStart structure.  Return a pointer to the </table> element, 
   // or to NULL if there is no </table>.
   //
   // The min and max size for column N (where the leftmost column has
   // N==1) is pStart->minW[1] and pStart->maxW[1].  The pStart->minW[0]
   // and pStart->maxW[0] entries contain the minimum and maximum widths
   // of the whole table, including any cell padding, cell spacing,
   // border width and "hspace".  The values of pStart->minW[I] for I>=1
   // do not contain any cell padding, cell spacing or border width.
   // Only pStart->minW[0] contains these extra spaces.
   //
   // The back references from </table>, </tr>, </td> and </th> back to
   // the <table> markup are also filled in.  And for each <td> and <th>
   // markup, the pTable and pEnd fields are set to their proper values.
   //
   // pStart    - The <table> markup
   // lineWidth - Total width available to the table

   TGHtmlElement *p;                   // Element being processed
   TGHtmlElement *pNext;               // Next element to process
   int iCol = 0;                      // Current column number.  1..N
   int iRow = 0;                      // Current row number
   TGHtmlElement *inRow = 0;           // Pointer to <TR>
   TGHtmlElement *inCol = 0;           // Pointer to <TD>
   int i, j;                          // Loop counters
   int n;                             // Number of columns
   int minW, maxW, requestedW;        // min, max, requested width for a cell
   int noWrap;                        // true for NOWRAP cells
   int colspan;                       // Column span for the current cell
   int rowspan;                       // Row span for the current cell
   char *z;                           // Value of a <table> parameter
   int cellSpacing;                   // Value of CELLSPACING parameter
   int cellPadding;                   // Value of CELLPADDING parameter
   int tbw;                           // Width of border around whole table
   int cbw;                           // Width of border around one cell
   int hspace;                        // Value of HSPACE parameter
   int separation;                    // Space between columns
   int margin;                        // Space between left margin and 1st col
   int availWidth;                    // Part of lineWidth still available
   int maxTableWidth;                 // Amount of lineWidth available to table
   int fromAbove[HTML_MAX_COLUMNS+1]; // Cell above extends thru this row
   int min0span[HTML_MAX_COLUMNS+1];  // Min for colspan=0 cells
   int max0span[HTML_MAX_COLUMNS+1];  // Max for colspan=0 cells
   int reqW[HTML_MAX_COLUMNS+1];      // Requested width for each column
   int hasbg;

   // colMin[A][B] is the absolute minimum width of all columns between
   // A+1 and B+1.  colMin[B][A] is the requested width of columns between
   // A+1 and B+1.  This information is used to add in the constraints imposed
   // by <TD COLSPAN=N> markup where N>=2.

   int colMin[HTML_MAX_COLUMNS+1][HTML_MAX_COLUMNS+1];
# define ColMin(A,B) colMin[(A)-1][(B)-1]
# define ColReq(A,B) colMin[(B)-1][(A)-1]
  
   if (pStart == 0 || pStart->type != Html_TABLE) return pStart;

   if (pStart->bgImage) pStart->hasbg = 1;

   TRACE_PUSH(HtmlTrace_Table1);
   TRACE(HtmlTrace_Table1, ("Starting TableDimensions... %s\n",
        pStart->MarkupArg("name", "")));

   pStart->nCol = 0;
   pStart->nRow = 0;

   z = pStart->MarkupArg("border", 0);
   if (z && *z == 0) z = "2";

   tbw = z ? atoi(z) : DFLT_BORDER;
   if (TableBorderMin && tbw < TableBorderMin) tbw = TableBorderMin;
   pStart->borderWidth = tbw;

   cbw = (tbw > 0);

   z = pStart->MarkupArg("cellpadding", 0);
   cellPadding = z ? atoi(z) : DFLT_CELLPADDING;
   cellSpacing = CellSpacing(pStart);

#ifdef DEBUG
   // The HtmlTrace_Table4 flag causes tables to be draw with borders
   // of 2, cellPadding of 5 and cell spacing of 2.  This makes the
   // table clearly visible.  Useful for debugging. */
   if (HtmlTraceMask & HtmlTrace_Table4) {
      tbw = pStart->borderWidth = 2;
      cbw = 1;
      cellPadding = 5;
      cellSpacing = 2;
      pStart->style.bgcolor = COLOR_Background;
   }
#endif

   separation = cellSpacing + 2 * (cellPadding + cbw);
   margin = tbw + cellSpacing + cbw + cellPadding;

   z = pStart->MarkupArg("hspace", 0);
   hspace = z ? atoi(z) : DFLT_HSPACE;

   // Figure out the maximum space available
   z = pStart->MarkupArg("width", 0);
   if (z) {
      int len = strlen(z);
      if (len > 0 && z[len-1] == '%') {
         maxTableWidth = (atoi(z) * lineWidth) / 100;
      } else {
         maxTableWidth = atoi(z);
      }
   } else {
      maxTableWidth = lineWidth;
   }
   maxTableWidth -= 2 * margin;
   SETMAX(maxTableWidth, 1);

   TRACE(HtmlTrace_Table1, ("lineWidth = %d, maxTableWidth = %d, margin = %d\n", 
         lineWidth, maxTableWidth, margin));

   for (p = pStart->pNext; p; p = pNext) {
      if (p->type == Html_EndTABLE) {
          ((TGHtmlRef *)p)->pOther = pStart;
         pStart->pEnd = p;
         break;
      }

      pNext = p->pNext;

      switch (p->type) {
         case Html_EndTD:
         case Html_EndTH:
         case Html_EndTABLE:
            ((TGHtmlRef *)p)->pOther = pStart;
            inCol = 0;
            break;

         case Html_EndTR:
            ((TGHtmlRef *)p)->pOther = pStart;
            inRow = 0;
            break;

         case Html_TR:
            ((TGHtmlRef *)p)->pOther = pStart;
            iRow++;
            pStart->nRow++;
            iCol = 0;
            inRow = p;
            availWidth = maxTableWidth;
            break;

         case Html_CAPTION:
            while (p && p->type != Html_EndTABLE 
                   && p->type != Html_EndCAPTION) p = p->pNext;
            break;

         case Html_TD:
         case Html_TH: {
            TGHtmlCell *cell = (TGHtmlCell *) p;
            inCol = p;
            if (!inRow) {
               // If the <TR> markup is omitted, insert it.
               TGHtmlElement *pNew = new TGHtmlRef(Html_TR, 1, 0, 0);
               if (pNew == 0) break;
               //pNew->type = Html_TR;
               pNew->count = 0;
               pNew->style = p->style;
               pNew->flags = p->flags;
               pNew->pNext = p;
               p->pPrev->pNext = pNew;
               p->pPrev = pNew;
               pNext = pNew;
               break;
            }
            do {
               iCol++;
            } while (iCol <= pStart->nCol && fromAbove[iCol] > iRow);
            cell->pTable = pStart;
            cell->pRow = inRow;
            colspan = cell->colspan;
            if (colspan == 0) colspan = 1;
            if (iCol + colspan - 1 > pStart->nCol) {
               int nCol = iCol + colspan - 1;
               if (nCol > HTML_MAX_COLUMNS) nCol = HTML_MAX_COLUMNS;
               for (i = pStart->nCol + 1; i <= nCol; i++) {
                  fromAbove[i] = 0;
                  pStart->minW[i] = 0;
                  pStart->maxW[i] = 0;
                  min0span[i] = 0;
                  max0span[i] = 0;
                  reqW[i] = 0;
                  for (j = 1; j < i; j++) {
                     ColMin(j,i) = 0;
                     ColReq(j,i) = 0;
                  }
               }
               pStart->nCol = nCol;
            }
            noWrap = (p->MarkupArg("nowrap", 0) != 0);
            hasbg = (pStart->hasbg || ((TGHtmlRef *)cell->pRow)->bgImage ||
                     cell->bgImage);
            pNext = MinMax(p, &minW, &maxW, availWidth, hasbg);
            cell->pEnd = pNext;
            requestedW = 0;
            if ((z = p->MarkupArg("width", 0)) != 0) {
               for (i = 0; isdigit(z[i]) || z[i] == '.'; i++) {}
               if (strcmp(z, "*") == 0) {
                  requestedW = availWidth;
               } else if (z[i] == 0) {
                  requestedW = atoi(z);
               } else if (z[i] == '%') {
                  requestedW = (atoi(z) * maxTableWidth + 99) / 100;
               }
            }

            TRACE(HtmlTrace_Table1,
                  ("Row %d Column %d: min=%d max=%d req=%d stop at %s\n",
                  iRow, iCol, minW, maxW, requestedW,
                  GetTokenName(((TGHtmlCell *)p)->pEnd)));

            if (noWrap) {
               if ((z = p->MarkupArg("rowspan", 0)) == 0) { // Hack ???
               //minW = (requestedW > 0 ? requestedW : maxW);
               } else {
                  minW = maxW;
               }
            }
            if (iCol + cell->colspan <= HTML_MAX_COLUMNS) {
               int min = 0;
               if (cell->colspan == 0) {
                  SETMAX(min0span[iCol], minW);
                  SETMAX(max0span[iCol], maxW);
                  min = min0span[iCol] + separation;
               } else if (colspan == 1) {
                  SETMAX(pStart->minW[iCol], minW);
                  SETMAX(pStart->maxW[iCol], maxW);
                  SETMAX(reqW[iCol], requestedW);
                  min = pStart->minW[iCol] + separation;
               } else {
                  int n = cell->colspan;
                  int per = maxW / n;
                  int ix;
                  SETMAX(ColMin(iCol, iCol+n-1), minW);
                  SETMAX(ColReq(iCol, iCol+n-1), requestedW);
                  min = minW + separation;
                  for (ix = iCol; ix < iCol + n; ix++) {
                     if (minW != maxW) {  //-- without this some tables are not displayed properly
                        SETMAX(pStart->maxW[ix], per);
                     }
                  }
               }
               availWidth -= min;
            }
            rowspan = cell->rowspan;
            if (rowspan == 0) rowspan = LARGE_NUMBER;
            if (rowspan > 1) {
               for (i = iCol; i < iCol + cell->colspan && i < HTML_MAX_COLUMNS; i++) {
                  fromAbove[i] = iRow + rowspan;
               }
            }
            if (cell->colspan > 1) {
               iCol += cell->colspan - 1;
            } else if (cell->colspan == 0) {
               iCol = HTML_MAX_COLUMNS + 1;
            }
            break;
         }
      }
   }

#ifdef DEBUG
   if (HtmlTraceMask & HtmlTrace_Table6) {
      char *zSpace = "";
      TRACE_INDENT;
      for (i = 1; i <= pStart->nCol; i++) {
         printf("%s%d:%d..%d", zSpace, i, pStart->minW[i], pStart->maxW[i]);
         if (reqW[i] > 0) {
            printf("(w=%d)", reqW[i]);
         }
         zSpace = "  ";
      }
      printf("\n");
      for (i = 1; i < pStart->nCol; i++) {
         for (j = i+1; j <= pStart->nCol; j++) {
            if (ColMin(i, j) > 0) {
               TRACE_INDENT;
               printf("ColMin(%d,%d) = %d\n", i, j, ColMin(i, j));
            }
            if (ColReq(i, j) > 0) {
               TRACE_INDENT;
               printf("ColReq(%d,%d) = %d\n", i, j, ColReq(i, j));
            }
         } 
      }
   }
#endif

   // Compute the min and max width of each column

   for (i = 1; i <= pStart->nCol; i++) {
      int sumMin, sumReq, sumMax;

      // Reduce the max[] field to N for columns that have "width=N"
      if (reqW[i] > 0) {
         pStart->maxW[i] = MAX(pStart->minW[i], reqW[i]);
      }

      // Expand the width of columns marked with "colspan=0". 

      if (min0span[i] > 0 || max0span[i] > 0) {
         int n = pStart->nCol - i + 1;
         minW = (min0span[i] + (n - 1) * (1 - separation)) / n;
         maxW = (max0span[i] + (n - 1) * (1 - separation)) / n;
         for (j = i; j <= pStart->nCol; j++) {
            SETMAX(pStart->minW[j], minW);
            SETMAX(pStart->maxW[j], maxW);
         }
      }

      // Expand the minW[] of columns to accomodate "colspan=N" constraints.
      // The minW[] is expanded up to the maxW[] first.  Then all the maxW[]s
      // are expanded in proportion to their sizes.  The same thing occurs
      // for reqW[]s.

      sumReq = reqW[i];
      sumMin = pStart->minW[i];
      sumMax = pStart->maxW[i];
      for (j = i-1; j >= 1; j--) {
         int cmin, creq;

         sumMin += pStart->minW[j];
         sumMax += pStart->maxW[j];
         sumReq += reqW[i];
         cmin = ColMin(j, i);

         if (cmin > sumMin) {
            int k;
            double scale;

            int *tminW = pStart->minW;
            int *tmaxW = pStart->maxW;
            if (sumMin < sumMax) {
               scale = (double) (cmin - sumMin) / (double) (sumMax - sumMin);
               for (k = j; k <= i; k++) {
                  sumMin -= tminW[k];
                  tminW[k] = (int) ((tmaxW[k] - tminW[k]) * scale + tminW[k]);
                  sumMin += tminW[k];
               }
            } else if (sumMin > 0) {
               scale = (double) cmin / (double) sumMin;
               for (k = j; k <= i; k++) {
                  sumMin -= tminW[k];
                  tminW[k] = tmaxW[k] = (int) (tminW[k] * scale);
                  sumMin += tminW[k];
               }
            } else {
               int unit = cmin / (i - j + 1);
               for (k = j; k <= i; k++) {
                  tminW[k] = tmaxW[k] = unit;
                  sumMin += tminW[k];
               }
            }
         }

         creq = ColReq(j, i);
         if (creq > sumReq) {
            int k;
            double scale;

            int *tmaxW = pStart->maxW;
            if (sumReq < sumMax) {
               scale = (double) (creq - sumReq) / (double) (sumMax - sumReq);
               for (k = j; k <= i; k++) {
                  sumReq -= reqW[k];
                  reqW[k] = (int) ((tmaxW[k] - reqW[k]) * scale + reqW[k]);
                  sumReq += reqW[k];
               }
            } else if (sumReq > 0) {
               scale = (double) creq / (double) sumReq;
               for (k = j; k <= i; k++) {
                  sumReq -= reqW[k];
                  reqW[k] = (int) (reqW[k] * scale);
                  sumReq += reqW[k];
               }
            } else {
               int unit = creq / (i - j + 1);
               for (k = j; k <= i; k++) {
                  reqW[k] = unit;
                  sumReq += reqW[k];
               }
            }
         }
      }
   }

#ifdef DEBUG
   if (HtmlTraceMask & HtmlTrace_Table6) {
      char *zSpace = "";
      TRACE_INDENT;
      for (i = 1; i <= pStart->nCol; i++) {
         printf("%s%d:%d..%d", zSpace, i, pStart->minW[i], pStart->maxW[i]);
         if (reqW[i] > 0) {
            printf("(w=%d)", reqW[i]);
         }
         zSpace = "  ";
      }
      printf("\n");
   }
#endif

   // Compute the min and max width of the whole table

   n = pStart->nCol;
   requestedW = tbw * 2 + (n + 1) * cellSpacing + n * 2 * (cellPadding + cbw);
   pStart->minW[0] = requestedW;
   pStart->maxW[0] = requestedW;
   for (i = 1; i <= pStart->nCol; i++) {
      pStart->minW[0] += pStart->minW[i];
      pStart->maxW[0] += pStart->maxW[i];
      requestedW += MAX(reqW[i], pStart->minW[i]);
   }

   // Possibly widen or narrow the table to accomodate a "width=" attribute
   z = pStart->MarkupArg("width", 0);
   if (z) {
      int len = strlen(z);
      int totalWidth;
      if (len > 0 && z[len-1] == '%') {
         totalWidth = (atoi(z) * lineWidth) / 100;
      } else {
         totalWidth = atoi(z);
      }
      SETMAX(totalWidth, pStart->minW[0]);
#if 1
      requestedW = totalWidth;
#else
      SETMAX(requestedW, totalWidth); //-- makes it too narrow
#endif
   }
   SETMAX(maxTableWidth, pStart->minW[0]);
   if (lineWidth && (requestedW > lineWidth)) {

      TRACE(HtmlTrace_Table5, ("RequestedW reduced to lineWidth: %d -> %d\n", 
            requestedW, lineWidth));

      requestedW = lineWidth;
   }
   if (requestedW > pStart->minW[0]) {
      float scale;
      int *tminW = pStart->minW;
      int *tmaxW = pStart->maxW;

      TRACE(HtmlTrace_Table5,
            ("Expanding table minW from %d to %d.  (reqW=%d width=%s)\n",
             tminW[0], requestedW, requestedW, z));

      if (tmaxW[0] > tminW[0]) {
         scale = (double) (requestedW - tminW[0]) / (double) (tmaxW[0] - tminW[0]);
         for (i = 1; i <= pStart->nCol; i++) {
            tminW[i] += (int) ((tmaxW[i] - tminW[i]) * scale);
            SETMAX(tmaxW[i], tminW[i]);
         }
      } else if (tminW[0] > 0) {
         scale = requestedW / (double) tminW[0];
         for (i = 1; i <= pStart->nCol; i++) {
            tminW[i] = (int) (tminW[i] * scale);
            tmaxW[i] = (int) (tmaxW[i] * scale);
         }
      } else if (pStart->nCol > 0) {
         int unit = (requestedW - margin) / pStart->nCol - separation;
         if (unit < 0) unit = 0;
         for (i = 1; i <= pStart->nCol; i++) {
            tminW[i] = tmaxW[i] = unit;
         }
      } else {
         tminW[0] = tmaxW[0] = requestedW;
      }
      pStart->minW[0] = requestedW;
      SETMAX(pStart->maxW[0], requestedW);
   }

#ifdef DEBUG
   if (HtmlTraceMask & HtmlTrace_Table5) {
      TRACE_INDENT;
      printf("Start with %s and ", GetTokenName(pStart));
      printf("end with %s\n", GetTokenName(p));
      TRACE_INDENT;
      printf("nCol=%d minWidth=%d maxWidth=%d\n",
      pStart->nCol, pStart->minW[0], pStart->maxW[0]);
      for (i = 1; i <= pStart->nCol; i++) {
         TRACE_INDENT;
         printf("Column %d minWidth=%d maxWidth=%d\n",
                i, pStart->minW[i], pStart->maxW[i]);
      }
   }
#endif

   TRACE(HtmlTrace_Table1,
         ("Result of TableDimensions: min=%d max=%d nCol=%d\n",
          pStart->minW[0], pStart->maxW[0], pStart->nCol));
   TRACE_POP(HtmlTrace_Table1);

   return p;
}

//______________________________________________________________________________
TGHtmlElement *TGHtml::MinMax(TGHtmlElement *p, int *pMin, int *pMax,
                             int /*lineWidth*/, int hasbg)
{
   // Given a list of elements, compute the minimum and maximum width needed
   // to render the list.  Stop the search at the first element seen that is
   // in the following set:
   //
   //       <tr>  <td>  <th>  </tr>  </td>  </th>  </table>
   //
   // Return a pointer to the element that stopped the search, or to NULL
   // if we ran out of data.
   //
   // Sometimes the value returned for both min and max will be larger than
   // the true minimum and maximum.  This is rare, and only occurs if the
   // element string contains figures with flow-around text.
   //
   //  p         - Start the search here
   //  pMin      - Return the minimum width here
   //  pMax      - Return the maximum width here
   //  lineWidth - Total width available

   int min = 0;             // Minimum width so far
   int max = 0;             // Maximum width so far
   int indent = 0;          // Amount of indentation (minimum)
   int obstacle = 0;        // Possible obstacles in the margin
   int x1 = 0;              // Length of current line assuming maximum length
   int x2 = 0;              // Length of current line assuming minimum length
   int x3 = 0;              // Like x1, but only within <PRE> tag
   int go = 1;              // Change to 0 to stop the loop
   int inpre = 0;           // Are we in <PRE>
   TGHtmlElement *pNext;     // Next element in the list
   int wstyle = 0;          // Current style for nowrap

   if (p->MarkupArg("nowrap", 0) != 0) {
      wstyle |= STY_NoBreak;
   }

   for (p = p->pNext; go && p; p = pNext) {
      pNext = p->pNext;
      if (!inpre) x3 = 0;
      switch (p->type) {
         case Html_PRE:
            inpre = 1;
            break;

         case Html_EndPRE:
            inpre = 0;
            break;

         case Html_Text: {
            TGHtmlTextElement *text = (TGHtmlTextElement *) p;
            x1 += text->w;
            x2 += text->w;
            SETMAX(max, x1);
            if (p->style.flags & STY_Preformatted) {
               x3 += text->w;
               SETMAX(min, x3);
            } else {
               SETMAX(min, x2);
            }
            break;
         }

         case Html_Space: {
            TGHtmlSpaceElement *space = (TGHtmlSpaceElement *) p;
            p->style.flags |= wstyle;
            if (p->style.flags & STY_Preformatted) {
               if (p->flags & HTML_NewLine) {
                  x1 = x2 = x3 = indent;
               } else {
                  x1 += space->w * p->count;
                  x2 += space->w * p->count;
                  x3 += space->w * p->count;
               }
            } else if (p->style.flags & STY_NoBreak) {
               if (x1 > indent) x1 += space->w;
               if (x2 > indent) x2 += space->w;
            } else {
               if (x1 > indent) x1 += space->w;
               x2 = indent;
            }
            break;
         }

         case Html_IMG: {
            TGHtmlImageMarkup *image = (TGHtmlImageMarkup *) p;
            switch (image->align) {
               case IMAGE_ALIGN_Left:
               case IMAGE_ALIGN_Right:
                  obstacle += image->w;
                  x1 = obstacle + indent;
                  x2 = indent;
                  SETMAX(min, x2);
                  SETMAX(min, image->w);
                  SETMAX(max, x1);
                  break;

               default:
                  x1 += image->w;
                  x2 += image->w;
                  if (p->style.flags & STY_Preformatted) {
                     SETMAX(min, x1);
                     SETMAX(max, x1);
                  } else {
                     SETMAX(min, x2);
                     SETMAX(max, x1);
                  }
                  break;
            }
            break;
         }

         case Html_TABLE: {
            TGHtmlTable *table = (TGHtmlTable *) p;
            /* pNext = TableDimensions(table, lineWidth - indent); */
            table->hasbg = hasbg;
            pNext = TableDimensions(table, 0);
            x1 = table->maxW[0] + indent + obstacle;
            x2 = table->minW[0] + indent;
            SETMAX(max, x1);
            SETMAX(min, x2);	
            x1 = indent + obstacle;
            x2 = indent;
            if (pNext && pNext->type == Html_EndTABLE) pNext = pNext->pNext;
            break;
         }

         case Html_UL:
         case Html_OL:
            indent += HTML_INDENT;
            x1 = indent + obstacle;
            x2 = indent;
            break;

         case Html_EndUL:
         case Html_EndOL:
            indent -= HTML_INDENT;
            if (indent < 0) indent = 0;
            x1 = indent + obstacle;
            x2 = indent;
            break;

         case Html_BLOCKQUOTE:
            indent += 2 * HTML_INDENT;
            x1 = indent + obstacle;
            x2 = indent;
            break;

         case Html_EndBLOCKQUOTE:
            indent -= 2 * HTML_INDENT;
            if (indent < 0) indent = 0;
            x1 = indent + obstacle;
            x2 = indent;
            break;

         case Html_APPLET:
         case Html_INPUT:
         case Html_SELECT:
         case Html_EMBED:
         case Html_TEXTAREA: {
            TGHtmlInput *input = (TGHtmlInput *) p;
            x1 += input->w + input->padLeft;
            if (p->style.flags & STY_Preformatted) {
               x3 += input->w + input->padLeft;
               SETMAX(min, x3);
               SETMAX(max, x1);
               x2 += input->w + input->padLeft;
            } else {
               SETMAX(min, indent + input->w);
               SETMAX(max, x1);
               x2 = indent;
            }
            break;
         }

         case Html_BR:
         case Html_P:
         case Html_EndP:
         case Html_DIV:
         case Html_EndDIV:
         case Html_H1:
         case Html_EndH1:
         case Html_H2:
         case Html_EndH2:
         case Html_H3:
         case Html_EndH3:
         case Html_H4:
         case Html_EndH4:
         case Html_H5:
         case Html_H6:
            x1 = indent + obstacle;
            x2 = indent;
            break;

         case Html_EndTD:
         case Html_EndTH:
         case Html_CAPTION:
         case Html_EndTABLE:
         case Html_TD:
         case Html_TR:
         case Html_TH:
         case Html_EndTR:
            go = 0;
            break;

         default:
            break;
      }

      if (!go) break;
   }

   *pMin = min;
   *pMax = max;

   return p;
}


// Vertical alignments: 

#define VAlign_Unknown    0
#define VAlign_Top        1
#define VAlign_Bottom     2
#define VAlign_Center     3
#define VAlign_Baseline   4


//______________________________________________________________________________
int TGHtmlMarkupElement::GetVerticalAlignment(int dflt)
{
   // Return the vertical alignment specified by the given element.

   char *z;
   int rc;

   z = MarkupArg("valign", 0);
   if (z == 0) {
      rc = dflt;
   } else if (strcasecmp(z, "top") == 0) {
      rc = VAlign_Top;
   } else if (strcasecmp(z, "bottom") == 0) {
      rc = VAlign_Bottom;
   } else if (strcasecmp(z, "center") == 0) {
      rc = VAlign_Center;
   } else if (strcasecmp(z, "baseline") == 0) {
      rc = VAlign_Baseline;
   } else{
      rc = dflt;
   }

   return rc;
}

//______________________________________________________________________________
TGHtmlElement *TGHtmlLayoutContext::TableLayout(TGHtmlTable *pTable)
{
   // Do all layout for a single table.  Return the </table> element or
   // NULL if the table is unterminated.

   TGHtmlElement *pEnd;        // The </table> element
   TGHtmlElement *p;           // For looping thru elements of the table
   TGHtmlElement *pNext;       // Next element in the loop
   TGHtmlElement *pCaption;    // Start of the caption text.  The <caption>
   TGHtmlElement *pEndCaption; // End of the caption.  The </caption>
   int width;               // Width of the table as drawn
   int cellSpacing;         // Value of cellspacing= parameter to <table>
   int cellPadding;         // Value of cellpadding= parameter to <table>
   int tbw;                 // Width of the 3D border around the whole table
   int cbw;                 // Width of the 3D border around a cell
   int pad;                 // cellPadding + borderwidth
   char *z;                 // A string
   int left_margin;         // The left edge of space available for drawing
   int lineWidth;           // Total horizontal space available for drawing
   int specWidth;           // Total horizontal drawing width per width= attr
   int separation;          // Distance between content of columns (or rows)
   int i;                   // Loop counter
   int n;                   // Number of columns
   int btm;                 // Bottom edge of previous row
   int iRow;                // Current row number
   int iCol;                // Current column number
   int colspan;             // Number of columns spanned by current cell
   int vspace;              // Value of the vspace= parameter to <table>
   int hspace;              // Value of the hspace= parameter to <table>
   int rowBottom;           // Bottom edge of content in the current row
   int defaultVAlign;       // Default vertical alignment for the current row
   char *zAlign;            // Value of the ALIGN= attribute of the <TABLE>
#define N (HTML_MAX_COLUMNS+1)
   int y[N];                // Top edge of each cell's content
   int x[N];                // Left edge of each cell's content
   int w[N];                // Width of each cell's content
   int ymax[N];             // Bottom edge of cell's content if valign=top
   TGHtmlElement *apElem[N]; // The <td> or <th> for each cell in a row
   int firstRow[N];         // First row on which a cell appears
   int lastRow[N];          // Row to which each cell span's
   int valign[N];           // Vertical alignment for each cell
   TGHtmlLayoutContext savedContext;  // Saved copy of the original pLC
   TGHtmlLayoutContext cellContext;   // Used to render a single cell
#ifdef TABLE_TRIM_BLANK
   extern int HtmlLineWasBlank;
#endif // TABLE_TRIM_BLANK

   if (pTable == 0 || pTable->type != Html_TABLE) return pTable;

   TRACE_PUSH(HtmlTrace_Table2);
   TRACE(HtmlTrace_Table2, ("Starting TableLayout() at %s\n", 
                          html->GetTokenName(pTable)));

   // Figure how much horizontal space is available for rendering 
   // this table.  Store the answer in lineWidth.  left_margin is
   // the left-most X coordinate of the table.  btm stores the top-most
   // Y coordinate.

   ComputeMargins(&left_margin, &btm, &lineWidth);

   TRACE(HtmlTrace_Table2, ("...btm=%d left=%d width=%d\n",
                           btm, left_margin, lineWidth));

   // figure out how much space the table wants for each column,
   // and in total..
   pEnd = html->TableDimensions(pTable, lineWidth);

   // If we don't have enough horizontal space to accomodate the minimum table
   // width, then try to move down past some obstruction (such as an
   // <IMG ALIGN=LEFT>) to give us more room.

   if (lineWidth < pTable->minW[0]) {
      WidenLine(pTable->minW[0], &left_margin, &btm, &lineWidth);

      TRACE(HtmlTrace_Table2, ("Widen to btm=%d left=%d width=%d\n", 
                             btm, left_margin, lineWidth));
   }
   savedContext = *this;
  
  // Figure out how wide to draw the table 
   z = pTable->MarkupArg("width", 0);
   if (z) {
      int len = strlen(z);
      if (len > 0 && z[len-1] == '%') {
         specWidth = (atoi(z) * lineWidth) / 100;
      } else {
         specWidth = atoi(z);
      }
   } else {
      specWidth = lineWidth;
   }
   if (specWidth < pTable->minW[0]) {
      width = pTable->minW[0];
   } else if (specWidth <= pTable->maxW[0]) {
      width = specWidth;
   } else {
      width = pTable->maxW[0];
   }

   // Compute the width and left edge position of every column in
   // the table

   z = pTable->MarkupArg("cellpadding", 0);
   cellPadding = z ? atoi(z) : DFLT_CELLPADDING;
   cellSpacing = html->CellSpacing(pTable);

   z = pTable->MarkupArg("vspace", 0);
   vspace = z ? atoi(z) : DFLT_VSPACE;

   z = pTable->MarkupArg("hspace", 0);
   hspace = z ? atoi(z) : DFLT_HSPACE;

#ifdef DEBUG
   if (HtmlTraceMask & HtmlTrace_Table4) {
      cellPadding = 5;
      cellSpacing = 2;
      if (vspace < 2) vspace = 2;
      if (hspace < 2) hspace = 2;
   }
#endif

   tbw = pTable->borderWidth;
   cbw = (tbw > 0);
   pad = cellPadding + cbw;
   separation = cellSpacing + 2 * pad;
   x[1] = left_margin + tbw + cellSpacing + pad;

   n = pTable->nCol;
   if (n <= 0 || pTable->maxW[0] <= 0) {
      // Abort if the table has no columns at all or if the total width
      // of the table is zero or less.
      return pEnd;
   }

   zAlign = pTable->MarkupArg("align", "");
   if (width <= lineWidth) {
      int align = pTable->style.align;
      if (align == ALIGN_Right || strcasecmp(zAlign, "right") == 0) {
         x[1] += lineWidth - width;
      } else if (align == ALIGN_Center && strcasecmp(zAlign, "left") != 0) {
         x[1] += (lineWidth - width) / 2;
      }
   }

   if (width == pTable->maxW[0]) {
      w[1] = pTable->maxW[1];
      for (i = 2; i <= n; i++) {
         w[i] = pTable->maxW[i];
         x[i] = x[i-1] + w[i-1] + separation;
      }
   } else if (width > pTable->maxW[0]) {
      int *tmaxW = pTable->maxW;
      double scale = ((double) width) / (double) tmaxW[0];
      w[1] = (int) (tmaxW[1] * scale);
      for (i = 2; i <= n; i++) {
         w[i] = (int) (tmaxW[i] * scale);
         x[i] = x[i-1] + w[i-1] + separation;
      }
   } else if (width > pTable->minW[0]) {
      float scale;
      int *tminW = pTable->minW;
      int *tmaxW = pTable->maxW;
      scale = (double) (width - tminW[0]) / (double) (tmaxW[0] - tminW[0]);
      w[1] = (int) (tminW[1] + (tmaxW[1] - tminW[1]) * scale);
      for (i = 2; i <= n; i++) {
         w[i] = (int) (tminW[i] + (tmaxW[i] - tminW[i]) * scale);
         x[i] = x[i-1] + w[i-1] + separation;
      }
   } else {
      w[1] = pTable->minW[1];
      for (i = 2; i <= n; i++) {
         w[i] = pTable->minW[i];
         x[i] = x[i-1] + w[i-1] + separation;
      }
   }
   w[n] = width - ((x[n] - x[1]) + 2 * (tbw + pad + cellSpacing));

   // Add notation to the pTable structure so that we will know where
   // to draw the outer box around the outside of the table.

   btm += vspace;
   pTable->y = btm;
   pTable->x = x[1] - (tbw + cellSpacing + pad);
   pTable->w = width;
   SETMAX(maxX, pTable->x + pTable->w);
   btm += tbw + cellSpacing;

   // Begin rendering rows of the table
   for (i = 1; i <= n; i++) {
      firstRow[i] = 0;
      lastRow[i] = 0;
      apElem[i] = 0;
   }
   p = pTable->pNext;
   rowBottom = btm;
   for (iRow = 1; iRow <= pTable->nRow; iRow++) {

      TRACE(HtmlTrace_Table2, ("Row %d: btm=%d\n",iRow,btm));

      // Find the start of the next row. Keep an eye out for the caption
      // while we search
      while (p && p->type != Html_TR) {
         if (p->type == Html_CAPTION) {
            pCaption = p;
            while (p && p != pEnd && p->type != Html_EndCAPTION) p = p->pNext;
            pEndCaption = p;
         }

         TRACE(HtmlTrace_Table3, ("Skipping token %s\n", html->GetTokenName(p)));

         p = p->pNext; 
      }
      if (p == 0) break;

      // Record default vertical alignment flag for this row
      defaultVAlign = p->GetVerticalAlignment(VAlign_Center);

      // Find every new cell on this row
      for (iCol = 1; iCol <= pTable->nCol && iCol <= HTML_MAX_COLUMNS; iCol++) {
         if (lastRow[iCol] < iRow) ymax[iCol] = 0;
      }
      iCol = 0;
      for (p = p->pNext; p && p->type != Html_TR && p != pEnd; p = pNext) {
         pNext = p->pNext;

         TRACE(HtmlTrace_Table3, ("Processing token %s\n", html->GetTokenName(p)));

         switch (p->type) {
            case Html_TD:
            case Html_TH:
               // Find the column number for this cell. Be careful to skip
               // columns which extend down to this row from prior rows
               do {
                  iCol++;
               } while (iCol <= HTML_MAX_COLUMNS && lastRow[iCol] >= iRow);

               TRACE(HtmlTrace_Table2,
                     ("Column %d: x=%d w=%d\n",iCol,x[iCol],w[iCol]));

               // Process the new cell. (Cells beyond the maximum number of
               // cells are simply ignored.)
               if (iCol <= HTML_MAX_COLUMNS) {
                  TGHtmlCell *cell = (TGHtmlCell *) p;
                  apElem[iCol] = p;
                  pNext = cell->pEnd;
                  if (cell->rowspan == 0) {
                     lastRow[iCol] = pTable->nRow;
                  } else {
                     lastRow[iCol] = iRow + cell->rowspan - 1;
                  }
                  firstRow[iCol] = iRow;

                  // Set vertical alignment flag for this cell
                  valign[iCol] = p->GetVerticalAlignment(defaultVAlign);

                  // Render cell contents and record the height
                  y[iCol] = btm + pad;
                  cellContext.html    = html;
                  cellContext.pStart = p->pNext;
                  cellContext.pEnd   = pNext;
                  cellContext.headRoom = 0;
                  cellContext.top    = y[iCol];
                  cellContext.bottom = y[iCol];
                  cellContext.left   = x[iCol];
                  cellContext.right  = 0;
                  cellContext.pageWidth = x[iCol] + w[iCol];
                  colspan = cell->colspan;
                  if (colspan == 0) {
                     for (i = iCol + 1;
                          i <= pTable->nCol && i <= HTML_MAX_COLUMNS; i++) {
                        cellContext.pageWidth += w[i] + separation;
                        lastRow[i] = lastRow[iCol];
                     }
                  } else if (colspan > 1) {
                     for (i = iCol + 1;
                          i < iCol + colspan && i <= HTML_MAX_COLUMNS; i++) {
                        cellContext.pageWidth += w[i] + separation;
                        lastRow[i] = lastRow[iCol];
                     }
                  }
                  cellContext.maxX = 0;
                  cellContext.maxY = 0;
                  cellContext.leftMargin = 0;
                  cellContext.rightMargin = 0;
                  cellContext.LayoutBlock();
#ifdef TABLE_TRIM_BLANK
                  // Cancel any trailing vertical whitespace caused
                  // by break markup
                  if (HtmlLineWasBlank) {
                     cellContext.maxY -= cellContext.headRoom;
                  }
#endif // TABLE_TRIM_BLANK
                  ymax[iCol] = cellContext.maxY;
                  SETMAX(ymax[iCol], y[iCol]);
                  cellContext.ClearMarginStack(&cellContext.leftMargin);
                  cellContext.ClearMarginStack(&cellContext.rightMargin);

                  // Set coordinates of the cell border
                  cell->x = x[iCol] - pad;
                  cell->y = btm;
                  cell->w = cellContext.pageWidth + 2 * pad - x[iCol];

                  TRACE(HtmlTrace_Table2,
                        ("Column %d top=%d bottom=%d h=%d left=%d w=%d\n",
                        iCol, y[iCol], ymax[iCol], ymax[iCol]-y[iCol], 
                        cell->x, cell->w));

                  // Advance the column counter for cells spaning multiple columns
                  if (colspan > 1) {
                     iCol += colspan - 1;
                  } else if (colspan == 0) {
                     iCol = HTML_MAX_COLUMNS + 1;
                  }
               }
               break;

         case Html_CAPTION:
            // Gotta remember where the caption is so we can render it
            // at the end
            pCaption = p;
            while (pNext && pNext != pEnd && pNext->type != Html_EndCAPTION) {
               pNext = pNext->pNext;
            }
            pEndCaption = pNext;
            break;
         }
      }

      // Figure out how high to make this row.
      for (iCol = 1; iCol <= pTable->nCol; iCol++) {
         if (lastRow[iCol] == iRow || iRow == pTable->nRow) {
            SETMAX(rowBottom, ymax[iCol]);
         }
      }

      TRACE(HtmlTrace_Table2, ("Total row height: %d..%d -> %d\n",
                             btm,rowBottom,rowBottom-btm));

      // Position every cell whose bottom edge ends on this row
      for (iCol = 1; iCol <= pTable->nCol; iCol++) {
         int dy;    // Extra space at top of cell used for vertical alignment
         TGHtmlCell *apCell = (TGHtmlCell *) apElem[iCol];

         // Skip any unused cells or cells that extend down thru 
         // subsequent rows
         if (apElem[iCol] == 0 ||
             (iRow != pTable->nRow && lastRow[iCol] > iRow)) continue;

            // Align the contents of the cell vertically.
            switch (valign[iCol]) {
               case VAlign_Unknown:
               case VAlign_Center:
                  dy = (rowBottom - ymax[iCol])/2;
                  break;
            case VAlign_Top:
            case VAlign_Baseline:
                  dy = 0;
                  break;
            case VAlign_Bottom:
                  dy = rowBottom - ymax[iCol];
                  break;
            }
            if (dy) {
               TGHtmlElement *pLast = apCell->pEnd;

               TRACE(HtmlTrace_Table3, ("Delta column %d by %d\n",iCol,dy));

               html->MoveVertically(apElem[iCol]->pNext, pLast, dy);
            }

            // Record the height of the cell so that the border can be drawn
            apCell->h = rowBottom + pad - apCell->y;
            apElem[iCol] = 0;
         }

         // Update btm to the height of the row we just finished setting
         btm = rowBottom + pad + cellSpacing;
      }

      btm += tbw;
      pTable->h = btm - pTable->y;
      SETMAX(maxY, btm);
      bottom = btm + vspace;

      // Render the caption, if there is one
      if (pCaption) {
      }

      // Whenever we do any table layout, we need to recompute all the 
      // TGHtmlBlocks. The following statement forces this.
      html->ResetBlocks(); // html->firstBlock = html->lastBlock = 0;

      // Adjust the context for text that wraps around the table, if
      // requested by an ALIGN=RIGHT or ALIGN=LEFT attribute.

      if (strcasecmp(zAlign, "left") == 0) {
         savedContext.maxX = maxX;
         savedContext.maxY = maxY;
         *this = savedContext;
         PushMargin(&leftMargin, pTable->w + 2, pTable->y + pTable->h + 2, 0);
      } else if (strcasecmp(zAlign, "right") == 0) {
         savedContext.maxX = maxX;
         savedContext.maxY = maxY;
         *this = savedContext;
         PushMargin(&rightMargin, pTable->w + 2, pTable->y + pTable->h + 2, 0);
      }

      // All done

      TRACE(HtmlTrace_Table2, (
            "Done with TableLayout().  x=%d y=%d w=%d h=%d Return %s\n",
            pTable->x, pTable->y, pTable->w, pTable->h,
            html->GetTokenName(pEnd)));
      TRACE_POP(HtmlTrace_Table2);

   return pEnd;
}


// Move all elements in the given list vertically by the amount dy
//
// p     - First element to move
// pLast - Last element.  Do move this one
// dy    - Amount by which to move

void TGHtml::MoveVertically(TGHtmlElement *p, TGHtmlElement *pLast, int dy) {

  if (dy == 0) return;

  while (p && p != pLast) {
    switch (p->type) {
      case Html_A:
        ((TGHtmlAnchor *)p)->y += dy;
        break;

      case Html_Text:
        ((TGHtmlTextElement *)p)->y += dy;
        break;

      case Html_LI:
        ((TGHtmlLi *)p)->y += dy;
        break;

      case Html_TD:
      case Html_TH:
        ((TGHtmlCell *)p)->y += dy;
        break;

      case Html_TABLE:
        ((TGHtmlTable *)p)->y += dy;
        break;

      case Html_IMG:
        ((TGHtmlImageMarkup *)p)->y += dy;
        break;

      case Html_INPUT:
      case Html_SELECT:
      case Html_APPLET:
      case Html_EMBED:
      case Html_TEXTAREA:
        ((TGHtmlInput *)p)->y += dy;
        break;

      default:
        break;
    }
    p = p->pNext;
  }
}
