/* @(#)root/base:$Name:  $:$Id: Gtypes.h,v 1.2 2000/06/13 12:24:55 brun Exp $ */

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
enum EMarkerStyle {kDot=1, kPlus, kStar, kCircle=4, kMultiply=5,
                   kFullDotSmall=6, kFullDotMedium=7, kFullDotlarge=8,
                   kFullCircle=20, kFullSquare=21, kFullTriangleUp=22,
                   kFullTriangleDown=23, kOpenCircle=24, kOpenSquare=25,
                   kOpenTriangleUp=26, kOpenDiamond=27, kOpenCross=28,
                   kFullStar=29, kOpenStar=30};
                   
#endif
