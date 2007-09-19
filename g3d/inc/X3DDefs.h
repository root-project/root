/* @(#)root/g3d:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_X3DDefs
#define ROOT_X3DDefs


/* Conditional compile for int math */

#ifdef USE_INTS

#define SHIFT   12
#define TRIG_ADJ        4096.0
typedef int     number;

#else

#define TRIG_ADJ        1.0
typedef float   number;

#endif

typedef struct POINT   point;
typedef struct SEGMENT segment;
typedef struct POLYGON polygon;
typedef struct COLOR_  Color;


typedef struct {
   short x, y;
} _XPoint;


struct POINT {
   int ClipFlags;
   int visibility;
   number x,y,z;
   float RX,BX,Y;
   _XPoint R;
   short sBX;
   float dist;
   int numSegs;
   segment **segs;
   int numPolys;
   polygon **polys;
   point   *redNext;
   point   *blueNext;
};

struct SEGMENT{
   point *P, *Q;
   Color *color;
   int numPolys;
   polygon **polys;
};

struct POLYGON{
   segment *m, *n;
   float minDist, maxDist;
   polygon *next;
   float dist;
   int   visibility;
   Color *color;
   int numPoints;
   point **points;
   int numSegs;
   segment **segs;
};

struct COLOR_{
   long value;
   long stereoColor;
   int  stipple;
   int red, green, blue;
};

#endif
