// @(#)root/asimage:$Id$
// Author: Valeriy Onuchin  20/04/2005

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/************************************************************************

Copyright 1987, 1998  The Open Group

All Rights Reserved.

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
OPEN GROUP BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Except as contained in this notice, the name of The Open Group shall not be
used in advertising or otherwise to promote the sale, use or other dealings
in this Software without prior written authorization from The Open Group.


Copyright 1987 by Digital Equipment Corporation, Maynard, Massachusetts.

                        All Rights Reserved

Permission to use, copy, modify, and distribute this software and its 
documentation for any purpose and without fee is hereby granted, 
provided that the above copyright notice appear in all copies and that
both that copyright notice and this permission notice appear in 
supporting documentation, and that the name of Digital not be
used in advertising or publicity pertaining to distribution of the
software without specific, written prior permission.  

DIGITAL DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL
DIGITAL BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
SOFTWARE.

************************************************************************/

#include "TPoint.h"

/*
 *     This file contains a few macros to help track
 *     the edge of a filled object.  The object is assumed
 *     to be filled in scanline order, and thus the
 *     algorithm used is an extension of Bresenham's line
 *     drawing algorithm which assumes that y is always the
 *     major axis.
 *     Since these pieces of code are the same for any filled shape,
 *     it is more convenient to gather the library in one
 *     place, but since these pieces of code are also in
 *     the inner loops of output primitives, procedure call
 *     overhead is out of the question.
 *     See the author for a derivation if needed.
 */


/*
 *  In scan converting polygons, we want to choose those pixels
 *  which are inside the polygon.  Thus, we add .5 to the starting
 *  x coordinate for both left and right edges.  Now we choose the
 *  first pixel which is inside the pgon for the left edge and the
 *  first pixel which is outside the pgon for the right edge.
 *  Draw the left pixel, but not the right.
 *
 *  How to add .5 to the starting x coordinate:
 *      If the edge is moving to the right, then subtract dy from the
 *  error term from the general form of the algorithm.
 *      If the edge is moving to the left, then add dy to the error term.
 *
 *  The reason for the difference between edges moving to the left
 *  and edges moving to the right is simple:  If an edge is moving
 *  to the right, then we want the algorithm to flip immediately.
 *  If it is moving to the left, then we don't want it to flip until
 *  we traverse an entire pixel.
 */

#define BRESINITPGON(dy, x1, x2, xStart, d, m, m1, incr1, incr2) { \
    int dx;\
\
    if ((dy) != 0) { \
        xStart = (x1); \
        dx = (x2) - xStart; \
        if (dx < 0) { \
            m = dx / (dy); \
            m1 = m - 1; \
            incr1 = -2 * dx + 2 * (dy) * m1; \
            incr2 = -2 * dx + 2 * (dy) * m; \
            d = 2 * m * (dy) - 2 * dx - 2 * (dy); \
        } else { \
            m = dx / (dy); \
            m1 = m + 1; \
            incr1 = 2 * dx - 2 * (dy) * m1; \
            incr2 = 2 * dx - 2 * (dy) * m; \
            d = -2 * m * (dy) + 2 * dx; \
        } \
    } \
}

#define BRESINCRPGON(d, minval, m, m1, incr1, incr2) { \
    if (m1 > 0) { \
        if (d > 0) { \
            minval += m1; \
            d += incr1; \
        } \
        else { \
            minval += m; \
            d += incr2; \
        } \
    } else {\
        if (d >= 0) { \
            minval += m1; \
            d += incr1; \
        } \
        else { \
            minval += m; \
            d += incr2; \
        } \
    } \
}


/*
 *     This structure contains all of the information needed
 *     to run the bresenham algorithm.
 *     The variables may be hardcoded into the declarations
 *     instead of using this structure to make use of
 *     register declarations.
 */
typedef struct {
    int minor_axis;	/* minor axis        */
    int d;		/* decision variable */
    int m, m1;		/* slope and slope+1 */
    int incr1, incr2;	/* error increments */
} BRESINFO;


#define BRESINITPGONSTRUCT(dmaj, min1, min2, bres) \
	BRESINITPGON(dmaj, min1, min2, bres.minor_axis, bres.d, \
                     bres.m, bres.m1, bres.incr1, bres.incr2)

#define BRESINCRPGONSTRUCT(bres) \
        BRESINCRPGON(bres.d, bres.minor_axis, bres.m, bres.m1, bres.incr1, bres.incr2)


/*
 *     These are the data structures needed to scan
 *     convert regions.  Two different scan conversion
 *     methods are available -- the even-odd method, and
 *     the winding number method.
 *     The even-odd rule states that a point is inside
 *     the polygon if a ray drawn from that point in any
 *     direction will pass through an odd number of
 *     path segments.
 *     By the winding number rule, a point is decided
 *     to be inside the polygon if a ray drawn from that
 *     point in any direction passes through a different
 *     number of clockwise and counter-clockwise path
 *     segments.
 *
 *     These data structures are adapted somewhat from
 *     the algorithm in (Foley/Van Dam) for scan converting
 *     polygons.
 *     The basic algorithm is to start at the top (smallest y)
 *     of the polygon, stepping down to the bottom of
 *     the polygon by incrementing the y coordinate.  We
 *     keep a list of edges which the current scanline crosses,
 *     sorted by x.  This list is called the Active Edge Table (AET)
 *     As we change the y-coordinate, we update each entry in 
 *     in the active edge table to reflect the edges new xcoord.
 *     This list must be sorted at each scanline in case
 *     two edges intersect.
 *     We also keep a data structure known as the Edge Table (ET),
 *     which keeps track of all the edges which the current
 *     scanline has not yet reached.  The ET is basically a
 *     list of ScanLineList structures containing a list of
 *     edges which are entered at a given scanline.  There is one
 *     ScanLineList per scanline at which an edge is entered.
 *     When we enter a new edge, we move it from the ET to the AET.
 *
 *     From the AET, we can implement the even-odd rule as in
 *     (Foley/Van Dam).
 *     The winding number rule is a little trickier.  We also
 *     keep the EdgeTableEntries in the AET linked by the
 *     nextWETE (winding EdgeTableEntry) link.  This allows
 *     the edges to be linked just as before for updating
 *     purposes, but only uses the edges linked by the nextWETE
 *     link as edges representing spans of the polygon to
 *     drawn (as with the even-odd rule).
 */

/*
 * for the winding number rule
 */
#define CLOCKWISE          1
#define COUNTERCLOCKWISE  -1 

typedef struct _EdgeTableEntry {
     int ymax;             /* ycoord at which we exit this edge. */
     BRESINFO bres;        /* Bresenham info to run the edge     */
     struct _EdgeTableEntry *next;       /* next in the list     */
     struct _EdgeTableEntry *back;       /* for insertion sort   */
     struct _EdgeTableEntry *nextWETE;   /* for winding num rule */
     int ClockWise;        /* flag for winding number rule       */
} EdgeTableEntry;


typedef struct _ScanLineList{
     int scanline;              /* the scanline represented */
     EdgeTableEntry *edgelist;  /* header node              */
     struct _ScanLineList *next;  /* next in the list       */
} ScanLineList;


typedef struct {
     int ymax;                 /* ymax for the polygon     */
     int ymin;                 /* ymin for the polygon     */
     ScanLineList scanlines;   /* header node              */
} EdgeTable;


/*
 * Here is a struct to help with storage allocation
 * so we can allocate a big chunk at a time, and then take
 * pieces from this heap when we need to.
 */
#define SLLSPERBLOCK 25

typedef struct _ScanLineListBlock {
     ScanLineList SLLs[SLLSPERBLOCK];
     struct _ScanLineListBlock *next;
} ScanLineListBlock;



/*
 *
 *     a few macros for the inner loops of the fill code where
 *     performance considerations don't allow a procedure call.
 *
 *     Evaluate the given edge at the given scanline.
 *     If the edge has expired, then we leave it and fix up
 *     the active edge table; otherwise, we increment the
 *     x value to be ready for the next scanline.
 *     The winding number rule is in effect, so we must notify
 *     the caller when the edge has been removed so he
 *     can reorder the Winding Active Edge Table.
 */
#define EVALUATEEDGEWINDING(pAET, pPrevAET, y, fixWAET) { \
   if (pAET->ymax == y) {          /* leaving this edge */ \
      pPrevAET->next = pAET->next; \
      pAET = pPrevAET->next; \
      fixWAET = 1; \
      if (pAET) \
         pAET->back = pPrevAET; \
   } \
   else { \
      BRESINCRPGONSTRUCT(pAET->bres); \
      pPrevAET = pAET; \
      pAET = pAET->next; \
   } \
}


/*
 *     Evaluate the given edge at the given scanline.
 *     If the edge has expired, then we leave it and fix up
 *     the active edge table; otherwise, we increment the
 *     x value to be ready for the next scanline.
 *     The even-odd rule is in effect.
 */
#define EVALUATEEDGEEVENODD(pAET, pPrevAET, y) { \
   if (pAET->ymax == y) {          /* leaving this edge */ \
      pPrevAET->next = pAET->next; \
      pAET = pPrevAET->next; \
      if (pAET) \
         pAET->back = pPrevAET; \
   } \
   else { \
      BRESINCRPGONSTRUCT(pAET->bres); \
      pPrevAET = pAET; \
      pAET = pAET->next; \
   } \
}

#define LARGE_COORDINATE 1000000
#define SMALL_COORDINATE -LARGE_COORDINATE

//______________________________________________________________________________
static void InsertEdgeInET(EdgeTable *ET, EdgeTableEntry *ETE, int scanline,
                           ScanLineListBlock **SLLBlock, int *iSLLBlock)
{
   //     Insert the given edge into the edge table.
   //    First we must find the correct bucket in the
   //    Edge table, then find the right slot in the
   //    bucket.  Finally, we can insert it.

   EdgeTableEntry *start, *prev;
   ScanLineList *pSLL, *pPrevSLL;
   ScanLineListBlock *tmpSLLBlock;

   /*
    * find the right bucket to put the edge into
    */
   pPrevSLL = &ET->scanlines;
   pSLL = pPrevSLL->next;
   while (pSLL && (pSLL->scanline < scanline)) {
      pPrevSLL = pSLL;
      pSLL = pSLL->next;
   }

    /*
     * reassign pSLL (pointer to ScanLineList) if necessary
     */
   if ((!pSLL) || (pSLL->scanline > scanline)) {
      if (*iSLLBlock > SLLSPERBLOCK-1) {
         tmpSLLBlock = new ScanLineListBlock;
         (*SLLBlock)->next = tmpSLLBlock;
         tmpSLLBlock->next = (ScanLineListBlock *)0;
         *SLLBlock = tmpSLLBlock;
         *iSLLBlock = 0;
      }
      pSLL = &((*SLLBlock)->SLLs[(*iSLLBlock)++]);

      pSLL->next = pPrevSLL->next;
      pSLL->edgelist = (EdgeTableEntry *)0;
      pPrevSLL->next = pSLL;
   }
   pSLL->scanline = scanline;

    /*
     * now insert the edge in the right bucket
     */
   prev = (EdgeTableEntry *)0;
   start = pSLL->edgelist;
   while (start && (start->bres.minor_axis < ETE->bres.minor_axis)) {
      prev = start;
      start = start->next;
   }
   ETE->next = start;

   if (prev) {
      prev->next = ETE;
   } else {
      pSLL->edgelist = ETE;
   }
}

//______________________________________________________________________________
static void CreateETandAET(int count, TPoint *pts, EdgeTable *ET, EdgeTableEntry *AET,
                           EdgeTableEntry *pETEs, ScanLineListBlock *pSLLBlock)
{
   //     This routine creates the edge table for
   //     scan converting polygons. 
   //     The Edge Table (ET) looks like:
   //
   //    EdgeTable
   //     --------
   //    |  ymax  |        ScanLineLists
   //    |scanline|-->------------>-------------->...
   //     --------   |scanline|   |scanline|
   //                |edgelist|   |edgelist|
   //                ---------    ---------
   //                    |             |
   //                    |             |
   //                    V             V
   //              list of ETEs   list of ETEs
   //
   //     where ETE is an EdgeTableEntry data structure,
   //     and there is one ScanLineList per scanline at
   //     which an edge is initially entered.

   TPoint *top, *bottom;
   TPoint *PrevPt, *CurrPt;
   int iSLLBlock = 0;
   int dy;

   if (count < 2)  return;

    /*
     *  initialize the Active Edge Table
     */
   AET->next = (EdgeTableEntry *)0;
   AET->back = (EdgeTableEntry *)0;
   AET->nextWETE = (EdgeTableEntry *)0;
   AET->bres.minor_axis = SMALL_COORDINATE;

    /*
     *  initialize the Edge Table.
     */
   ET->scanlines.next = (ScanLineList *)0;
   ET->ymax = SMALL_COORDINATE;
   ET->ymin = LARGE_COORDINATE;
   pSLLBlock->next = (ScanLineListBlock *)0;

   PrevPt = &pts[count-1];

    /*
     *  for each vertex in the array of points.
     *  In this loop we are dealing with two vertices at
     *  a time -- these make up one edge of the polygon.
     */
   while (count--) {
      CurrPt = pts++;

        /*
         *  find out which point is above and which is below.
         */
      if (PrevPt->fY > CurrPt->fY) {
         bottom = PrevPt, top = CurrPt;
         pETEs->ClockWise = 0;
      } else {
         bottom = CurrPt, top = PrevPt;
         pETEs->ClockWise = 1;
      }

        /*
         * don't add horizontal edges to the Edge table.
         */
      if (bottom->fY != top->fY) {
         pETEs->ymax = bottom->fY-1;  /* -1 so we don't get last scanline */

            /*
             *  initialize integer edge algorithm
             */
         dy = bottom->fY - top->fY;
         BRESINITPGONSTRUCT(dy, top->fX, bottom->fX, pETEs->bres);

         InsertEdgeInET(ET, pETEs, top->fY, &pSLLBlock, &iSLLBlock);

	      if (PrevPt->fY > ET->ymax) ET->ymax = PrevPt->fY;
	      if (PrevPt->fY < ET->ymin) ET->ymin = PrevPt->fY;
         pETEs++;
      }
      PrevPt = CurrPt;
   }
}

//______________________________________________________________________________
static void loadAET(EdgeTableEntry *AET, EdgeTableEntry *ETEs)
{
   //    This routine moves EdgeTableEntries from the
   //     EdgeTable into the Active Edge Table,
   //     leaving them sorted by smaller x coordinate.

   EdgeTableEntry *pPrevAET;
   EdgeTableEntry *tmp;

   pPrevAET = AET;
   AET = AET->next;
   while (ETEs) {
      while (AET && (AET->bres.minor_axis < ETEs->bres.minor_axis))  {
         pPrevAET = AET;
         AET = AET->next;
      }
      tmp = ETEs->next;
      ETEs->next = AET;
      if (AET) {
         AET->back = ETEs;
      }
      ETEs->back = pPrevAET;
      pPrevAET->next = ETEs;
      pPrevAET = ETEs;

      ETEs = tmp;
   }
}

//______________________________________________________________________________
static int InsertionSort(EdgeTableEntry * AET)
{
   // InsertionSort
   //
   //    Just a simple insertion sort using
   //     pointers and back pointers to sort the Active
   //     Edge Table.

   EdgeTableEntry *pETEchase;
   EdgeTableEntry *pETEinsert;
   EdgeTableEntry *pETEchaseBackTMP;
   int changed = 0;

   AET = AET->next;
   while (AET) {
      pETEinsert = AET;
      pETEchase = AET;
      while (pETEchase->back->bres.minor_axis > AET->bres.minor_axis) {
         pETEchase = pETEchase->back;
      }

      AET = AET->next;
      if (pETEchase != pETEinsert) {
         pETEchaseBackTMP = pETEchase->back;
         pETEinsert->back->next = AET;
         if (AET) {
            AET->back = pETEinsert->back;
         }
         pETEinsert->next = pETEchase;
         pETEchase->back->next = pETEinsert;
         pETEchase->back = pETEinsert;
         pETEinsert->back = pETEchaseBackTMP;
         changed = 1;
      }
   }
   return (changed);
}

//______________________________________________________________________________
static void FreeStorage(ScanLineListBlock *pSLLBlock)
{
   // Clean up our act.

   ScanLineListBlock *tmpSLLBlock;

   while (pSLLBlock) {
      tmpSLLBlock = pSLLBlock->next;
      delete pSLLBlock;
      pSLLBlock = tmpSLLBlock;
   }
}

