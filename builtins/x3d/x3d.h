/* @(#)root/x3d:$Id$ */
/* Author: Mark Spychalla*/

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_x3d
#define ROOT_x3d

/*
  Copyright 1992 Mark Spychalla

  Permission to use, copy, modify, distribute, and sell this software and
  its documentation for any purpose is hereby granted without fee,
  provided that the above copyright notice appear in all copies and that
  both that copyright notice and this permission notice appear in
  supporting documentation, and that the name of Mark Spychalla not be used
  in advertising or publicity pertaining to distribution of the software
  without specific, written prior permission.  Mark Spychalla makes no
  representations about the suitability of this software for any purpose.
  It is provided "as is" without express or implied warranty.

  Mark Spychalla disclaims all warranties with regard to this software,
  including all implied warranties of merchantability and fitness, in no
  event shall Mark Spychalla be liable for any special, indirect or
  consequential damages or any damages whatsoever resulting from loss of use,
  data or profits, whether in an action of contract, negligence or other
  tortious action, arising out of or in connection with the use or performance
  of this software.
*/

#include "X3DDefs.h"

#include <X11/Xlib.h>


/* Constants */

#define RClipWithRight   6
#define RClipWithLeft    5
#define PointBehind      4
#define BClipWithRight   3
#define BClipWithLeft    2
#define ClipWithRight    3
#define ClipWithLeft     2
#define ClipWithBottom   1
#define ClipWithTop      0

#define RRight     (1 << RClipWithRight)
#define RLeft      (1 << RClipWithLeft)
#define Behind     (1 << PointBehind)
#define BRight     (1 << BClipWithRight)
#define BLeft      (1 << BClipWithLeft)
#define Right      (1 << ClipWithRight)
#define Left       (1 << ClipWithLeft)
#define Bottom     (1 << ClipWithBottom)
#define Top        (1 << ClipWithTop)
#define Bmask      (~BRight & ~BLeft)
#define Rmask      (~RRight & ~RLeft)
#define RBmask     (Rmask & Bmask)
#define RLeftRight (RRight | RLeft)
#define ALLmask    (RRight | RLeft | Behind | BRight | BLeft | Bottom | Top )

#define NUMBOUNDS       8
#define NUMSTIPPLES     17
#define MAXVALUE        6
#define VALUESCALE      51
#define MAXCOLORDIST    (443.40501)
#define STIPPLESIZE     4
#define BITSPERBYTE     8
#define MAXCOLORS       256
#define MAXLINE         8192
#define MAXOPTIONLEN    256
#define TMPSTRLEN       16
#define SMALLMOVEMENT   40000
#define POINTERRATIO    0.007
#define MARGIN          30
#define TWOPI           6.2831853
#define REQUESTFACTOR   3
#define EIGHTBIT        8
#define POSTSCRIPT      1
#define HPGL            0
#define HELPLINES       40

#define MAXSTACK    100
#define STOP        10

#define EOK             0
#define ERROR           -1

#define FONT         "9x15"
#define TITLEFONT    "12x24"
#define BOLDFONT     "9x15bold"
#define FIXED        "fixed"

#define LONGESTSTRING "     ROTATE OBJECT ABOUT Z   Horizontal   "

/* Color Modes */

#define BW      1
#define STEREO  2
#define COLOR   3

/* Rendering modes */

#define WIREFRAME       1
#define HIDDENLINE      2
#define SOLID           3

/* Supported Depths */

#define ONE             1
#define EIGHT           8

/* Double buffering constants */

#define MAX_COLORS           232
#define BUFFER_CMAP          11
#define BUFFER0              240
#define BUFFER1              15

/* Segment intersection constants */

#define ENDS_INTERSECT   3
#define SAME             2
#define ABOVE            1
#define INTERSECT        0
#define BELOW           -1

/* x3d macros */


#define clipWithBottom(x,y,dx,dy,V)    { x -= ((dx * (V+y)) / dy); y = -V; }
#define clipWithTop(x,y,dx,dy,V)       { x += ((dx * (V-y)) / dy); y =  V; }
#define clipWithLeftSide(x,y,dx,dy,H)  { y -= ((dy * (H+x)) / dx); x = -H; }
#define clipWithRightSide(x,y,dx,dy,H) { y += ((dy * (H-x)) / dx); x =  H; }

#define FONTHEIGHT(font) (font->ascent + font->descent)

#define HelpPrint(g, x, y, string){                                        \
   XDrawString(g->dpy, g->helpWin, g->helpGc, x, y, string, strlen(string)); \
   y += FONTHEIGHT(g->font);                                               \
}

#define swapPtrs(ptr1, ptr2)                                         \
       ptr1 = (polygon **)((long)ptr1 ^ (long)ptr2);                 \
       ptr2 = (polygon **)((long)ptr2 ^ (long)ptr1);                 \
       ptr1 = (polygon **)((long)ptr1 ^ (long)ptr2);

#define median5(v1,v2,v3,v4,v5)                                      \
   if((*v1)->dist < (*v2)->dist){                                    \
      swapPtrs(v1,v2)                                                \
      }                                                              \
   if((*v3)->dist < (*v4)->dist){                                    \
      swapPtrs(v3,v4)                                                \
      }                                                              \
   if((*v1)->dist < (*v3)->dist){                                    \
      swapPtrs(v1,v3)                                                \
      swapPtrs(v2,v4)                                                \
      }                                                              \
   if((*v2)->dist < (*v5)->dist){                                    \
      swapPtrs(v2,v5)                                                \
      }                                                              \
   if((*v2)->dist < (*v3)->dist){                                    \
      swapPtrs(v2,v3)                                                \
      swapPtrs(v4,v5)                                                \
      }                                                              \
   if((*v3)->dist < (*v5)->dist){                                    \
      swapPtrs(v3,v5)                                                \
      }



/* Types */



typedef struct STACKELEMENT{
   int start, end;
} StackElement;


typedef struct XSEGMENT{
   _XPoint P, Q;
} xsegment;


typedef struct ANGLEPOINT{
   double x, y, z;
} anglePoint;

typedef struct OINFO{

/* Geometry information */

point   *points;
segment *segs;
polygon *polys;
polygon **list;

/* Clipping information */

point *bounds;
int  objClip;
int  Hmin, Vmin, Hmax, Vmax;
int  Hmin1, Vmin1, Hmax1, Vmax1;
int  Hmin2, Vmin2, Hmax2, Vmax2;
int  copyX, copyY, copyWidth, copyHeight;
int  fillX, fillY, fillWidth, fillHeight;

int numPoints, numSegs, numPolys;

/* Position information */

float tX, tY, tZ, dtX, dtY, dtZ;
float oX, oY, oZ, doX, doY, doZ;
double X, Y, Z,dX, dY, dZ;
double focus, scale, dscale;
float  BViewpointX, viewpointY;

} Oinfo;

typedef struct GINFO{

/* Position variables */

int dpyX, dpyY;
int winX, winY, helpWinX, helpWinY;
int oldPointerX, oldPointerY;

/* Font variables */

XFontStruct *font, *titleFont, *boldFont;

/* flags */

int depth, renderMode, buffer, mono, stereo, stereoBlue;
int ColorSelect, Block, Relative, helpMenu, modeChanged;

/* Color variables */

Color *colors;
int   numColors;
long stereoBlack, redMask, blueMask;

/* X window variables */

XSegment *redSegments, *blueSegments;
long *redColors;
int numberBlue, numberRed, winH, winV;
int numRedColors;
int requestSize;
long black, white, Black, Red, Blue, Purple;

polygon *edgeList;
polygon *freeList;

Window   win, helpWin;
Display  *dpy;
Drawable dest;
GC       gc, helpGc;
Colormap colormap;
long red, blue, mask;
XColor cmapColors[3][256];
XColor wireframeColors[2][256];
char *DisplayName, *Geometry;
Pixmap stipple[NUMSTIPPLES], pix;
} Ginfo;


#endif
