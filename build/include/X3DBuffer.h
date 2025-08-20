/* @(#)root/g3d:$Id$ */
/* Author: Nenad Buncic   13/12/95*/

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_X3DBuffer
#define ROOT_X3DBuffer

typedef struct _x3d_data_ {
   int  numPoints;
   int  numSegs;
   int  numPolys;
   float *points; /* x0, y0, z0, x1, y1, z1, ..... ..... ....    */
   int *segs;     /* c0, p0, q0, c1, p1, q1, ..... ..... ....    */
   int *polys;    /* c0, n0, s0, s1, ... sn, c1, n1, s0, ... sn  */
} X3DBuffer;

typedef struct _x3d_sizeof_ {
   int  numPoints;
   int  numSegs;
   int  numPolys;
} Size3D;

#ifdef __cplusplus
extern "C" int AllocateX3DBuffer ();
extern "C" void FillX3DBuffer (X3DBuffer *buff);
extern "C" Size3D* gFuncSize3D();
#else
extern int AllocateX3DBuffer ();
extern void FillX3DBuffer (X3DBuffer *buff);
extern Size3D* gFuncSize3D();
#endif

#define gSize3D (*gFuncSize3D())

#endif
