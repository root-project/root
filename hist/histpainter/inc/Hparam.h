/* @(#)root/histpainter:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Hparam
#define ROOT_Hparam


////////////////////////////////////////////////////////////////////////////////
/*! \struct Hparam_t
    \brief Histogram parameters structure.

Structure to store current histogram's parameters.

Used internally by THistPainter to manage histogram parameters.

*/

#include "RtypesCore.h"

typedef struct Hparam_t {
   Double_t  xbinsize;      ///< Bin size in case of equidistant bins
   Double_t  xlowedge;      ///< Low edge of axis
   Double_t  xmin;          ///< Minimum value along X
   Double_t  xmax;          ///< Maximum value along X
   Double_t  ybinsize;      ///< Bin size in case of equidistant bins
   Double_t  ylowedge;      ///< Low edge of axis
   Double_t  ymin;          ///< Minimum value along y
   Double_t  ymax;          ///< Maximum value along y
   Double_t  zbinsize;      ///< Bin size in case of equidistant bins
   Double_t  zlowedge;      ///< Low edge of axis
   Double_t  zmin;          ///< Minimum value along Z
   Double_t  zmax;          ///< Maximum value along Z
   Double_t  factor;        ///< Multiplication factor (normalization)
   Double_t  allchan;       ///< Integrated sum of contents
   Double_t  baroffset;     ///< Offset of bin for bars or legos [0,1]
   Double_t  barwidth;      ///< Width of bin for bars and legos [0,1]
   Int_t     xfirst;        ///< First bin number along X
   Int_t     xlast;         ///< Last bin number along X
   Int_t     yfirst;        ///< First bin number along Y
   Int_t     ylast;         ///< Last bin number along Y
   Int_t     zfirst;        ///< First bin number along Z
   Int_t     zlast;         ///< Last bin number along Z
} Hparam_t;

#endif
