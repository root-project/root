/* @(#)root/base:$Name:  $:$Id: Gtypes.h,v 1.4 2001/10/23 13:22:41 brun Exp $ */

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

typedef short     Font_t;        //Font number (short)
typedef short     Style_t;       //Style number (short)
typedef short     Marker_t;      //Marker number (short)
typedef short     Width_t;       //Line width (short)
typedef short     Color_t;       //Color number (short)
typedef short     SCoord_t;      //Screen coordinates (short)
typedef double    Coord_t;       //Pad world coordinates (double)
typedef float     Angle_t;       //Graphics angle (float)
typedef float     Size_t;        //Attribute size (float)

enum EColor { kWhite, kBlack, kRed, kGreen, kBlue, kYellow, kMagenta, kCyan };
enum ELineStyle { kSolid = 1, kDashed, kDotted, kDashDotted };
enum EMarkerStyle {kDot=1, kPlus, kStar, kCircle=4, kMultiply=5,
                   kFullDotSmall=6, kFullDotMedium=7, kFullDotLarge=8,
                   kOpenTriangleDown = 16, kFullCross= 18,
                   kFullCircle=20, kFullSquare=21, kFullTriangleUp=22,
                   kFullTriangleDown=23, kOpenCircle=24, kOpenSquare=25,
                   kOpenTriangleUp=26, kOpenDiamond=27, kOpenCross=28,
                   kFullStar=29, kOpenStar=30};
                   
#endif
