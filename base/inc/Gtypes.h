/* @(#)root/base:$Name:  $:$Id: Gtypes.h,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Gtypes
#define ROOT_Gtypes


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Gtypes                                                               //
//                                                                      //
// Types used by the graphics classes.                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Htypes
#include "Htypes.h"
#endif

typedef short     Font_t;        //Font number
typedef short     Style_t;       //Style number
typedef short     Marker_t;      //Marker number
typedef short     Width_t;       //Line width
typedef short     Color_t;       //Color number
typedef short     SCoord_t;      //Screen coordinates
typedef double    Coord_t;       //Pad world coordinates
typedef float     Angle_t;       //Graphics angle
typedef float     Size_t;        //Attribute size

enum EColor { kWhite, kBlack, kRed, kGreen, kBlue, kYellow, kMagenta, kCyan };
enum ELineStyle { kSolid = 1, kDashed, kDotted, kDashDotted };

#endif
