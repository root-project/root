// @(#)root/graf:$Name$:$Id$
// Author: Rene Brun   23/02/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPoints
#define ROOT_TPoints


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPoints                                                              //
//                                                                      //
// 2-D graphics point (world coordinates, i.e. floating point).         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Gtypes
#include "Gtypes.h"
#endif


class TPoints {

private:
   Coord_t    fX;           //X world coordinate
   Coord_t    fY;           //Y world coordinate

public:
   TPoints() : fX(0), fY(0) { }
   TPoints(Coord_t xy) : fX(xy), fY(xy) { }
   TPoints(Coord_t x, Coord_t y) : fX(x), fY(y) { }
   virtual ~TPoints() { }
   Coord_t    GetX() const { return fX; }
   Coord_t    GetY() const { return fY; }
   void       SetX(Coord_t x) { fX = x; }
   void       SetY(Coord_t y) { fY = y; }

   ClassDef(TPoints,0)  //2-D graphics point
};

#endif
