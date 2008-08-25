// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveFrameBox.h"
#include "TColor.h"

//==============================================================================
// TEveFrameBox
//==============================================================================

//______________________________________________________________________________
//
// Description of a 2D or 3D frame that can be used to visually group
// a set of objects.

ClassImp(TEveFrameBox);

//______________________________________________________________________________
TEveFrameBox::TEveFrameBox() :
   fFrameType   (kFT_None),
   fFrameSize   (0),
   fFramePoints (0),

   fFrameWidth  (1),
   fFrameColor  (1),
   fBackColor   (0),
   fFrameFill   (kFALSE),
   fDrawBack    (kFALSE)
{
   // Default constructor.

   fFrameRGBA[0] = fFrameRGBA[1] = fFrameRGBA[2] = 0;   fFrameRGBA[3] = 255;
   fBackRGBA [0] = fBackRGBA [1] = fBackRGBA [2] = 255; fBackRGBA [3] = 255;
}

//______________________________________________________________________________
TEveFrameBox::~TEveFrameBox()
{
   // Destructor.

   delete [] fFramePoints;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveFrameBox::SetAAQuadXY(Float_t x,  Float_t y, Float_t z,
                               Float_t dx, Float_t dy)
{
   // Setup for axis-aligned rectangle with one corner at x, y, z and
   // given sizes in x (dx) and y (dy).

   fFrameType = kFT_Quad;
   fFrameSize = 12;
   delete [] fFramePoints;
   fFramePoints = new Float_t [fFrameSize];
   Float_t* p = fFramePoints;
   p[0] = x;    p[1] = y;    p[2] = z; p += 3;
   p[0] = x+dx; p[1] = y;    p[2] = z; p += 3;
   p[0] = x+dx; p[1] = y+dy; p[2] = z; p += 3;
   p[0] = x ;   p[1] = y+dy; p[2] = z; p += 3;
}

//______________________________________________________________________________
void TEveFrameBox::SetAAQuadXZ(Float_t x,  Float_t y, Float_t z,
                               Float_t dx, Float_t dz)
{
   // Setup for axis-aligned rectangle with one corner at x, y, z and
   // given sizes in x (dx) and z (dz).

   fFrameType = kFT_Quad;
   fFrameSize = 12;
   delete [] fFramePoints;
   fFramePoints = new Float_t [fFrameSize];
   Float_t* p = fFramePoints;
   p[0] = x;    p[1] = y; p[2] = z;    p += 3;
   p[0] = x+dx; p[1] = y; p[2] = z;    p += 3;
   p[0] = x+dx; p[1] = y; p[2] = z+dz; p += 3;
   p[0] = x ;   p[1] = y; p[2] = z+dz; p += 3;
}

//______________________________________________________________________________
void TEveFrameBox::SetQuadByPoints(const Float_t* pointArr, Int_t nPoints)
{
   // Setup frame with explicitly given corner coordinates.
   // Arguments:
   //   pointArr - array containing the 3D points
   //   nPoint   - number of points, size of array divided by 3

   fFrameType = kFT_Quad;
   fFrameSize = 3*nPoints;
   delete [] fFramePoints;
   fFramePoints = new Float_t [fFrameSize];
   memcpy(fFramePoints, pointArr, fFrameSize*sizeof(Float_t));
}

//______________________________________________________________________________
void TEveFrameBox::SetAABox(Float_t x,  Float_t y,  Float_t z,
                            Float_t dx, Float_t dy, Float_t dz)
{
   // Setup for axis-aligned box with one corner at x, y, z and
   // given sizes in x (dx), y (dy) and z (dz).

   fFrameType = kFT_Box;
   fFrameSize = 24;
   delete [] fFramePoints;
   fFramePoints = new Float_t [fFrameSize];

   Float_t* p = fFramePoints;
   //bottom
   p[0] = x;       p[1] = y + dy;  p[2] = z;       p += 3;
   p[0] = x + dx;  p[1] = y + dy;  p[2] = z;       p += 3;
   p[0] = x + dx;  p[1] = y;       p[2] = z;       p += 3;
   p[0] = x;       p[1] = y;       p[2] = z;       p += 3;
   //top
   p[0] = x;       p[1] = y + dy;  p[2] = z + dz;  p += 3;
   p[0] = x + dx;  p[1] = y + dy;  p[2] = z + dz;  p += 3;
   p[0] = x + dx;  p[1] = y;       p[2] = z + dz;  p += 3;
   p[0] = x;       p[1] = y;       p[2] = z + dz;
}

//______________________________________________________________________________
void TEveFrameBox::SetAABoxCenterHalfSize(Float_t x,  Float_t y,  Float_t z,
                                          Float_t dx, Float_t dy, Float_t dz)
{
   // Setup for axis-aligned box with center at x, y, z and given
   // half-sizes in x (dx), y (dy) and z (dz).

   fFrameType = kFT_Box;
   fFrameSize = 24;
   delete [] fFramePoints;
   fFramePoints = new Float_t [fFrameSize];

   Float_t* p = fFramePoints;
   //bottom
   p[0] = x - dx;  p[1] = y + dy;  p[2] = z - dz;  p += 3;
   p[0] = x + dx;  p[1] = y + dy;  p[2] = z - dz;  p += 3;
   p[0] = x + dx;  p[1] = y - dy;  p[2] = z - dz;  p += 3;
   p[0] = x - dx;  p[1] = y - dy;  p[2] = z - dz;  p += 3;
   //top
   p[0] = x - dx;  p[1] = y + dy;  p[2] = z + dz;  p += 3;
   p[0] = x + dx;  p[1] = y + dy;  p[2] = z + dz;  p += 3;
   p[0] = x + dx;  p[1] = y - dy;  p[2] = z + dz;  p += 3;
   p[0] = x - dx;  p[1] = y - dy;  p[2] = z + dz;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveFrameBox::SetFrameColor(Color_t ci)
{
   // Set color of the frame.

   fFrameColor = ci;
   TEveUtil::ColorFromIdx(ci, fFrameRGBA, kTRUE);
}

//______________________________________________________________________________
void TEveFrameBox::SetFrameColorPixel(Pixel_t pix)
{
   // Set color of the frame.

   SetFrameColor(Color_t(TColor::GetColor(pix)));
}

//______________________________________________________________________________
void TEveFrameBox::SetFrameColorRGBA(UChar_t r, UChar_t g, UChar_t b, UChar_t a)
{
   // Set color of the frame.

   fFrameColor = Color_t(TColor::GetColor(r, g, b));
   fFrameRGBA[0] = r;
   fFrameRGBA[1] = g;
   fFrameRGBA[2] = b;
   fFrameRGBA[3] = a;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveFrameBox::SetBackColor(Color_t ci)
{
   // Set color of the background polygon.

   fBackColor = ci;
   TEveUtil::ColorFromIdx(ci, fBackRGBA, kTRUE);
}

//______________________________________________________________________________
void TEveFrameBox::SetBackColorPixel(Pixel_t pix)
{
   // Set color of the background polygon.

   SetBackColor(Color_t(TColor::GetColor(pix)));
}

//______________________________________________________________________________
void TEveFrameBox::SetBackColorRGBA(UChar_t r, UChar_t g, UChar_t b, UChar_t a)
{
   // Set color of the background polygon.

   fBackColor = Color_t(TColor::GetColor(r, g, b));
   fBackRGBA[0] = r;
   fBackRGBA[1] = g;
   fBackRGBA[2] = b;
   fBackRGBA[3] = a;
}
