// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveBox.h"


//______________________________________________________________________________
// Description of TEveBox
//

ClassImp(TEveBox);

//______________________________________________________________________________
TEveBox::TEveBox(const char* n, const char* t) :
   TEveShape(n, t)
{
   // Constructor.
}

//______________________________________________________________________________
TEveBox::~TEveBox()
{
   // Destructor.
}

//______________________________________________________________________________
void TEveBox::SetVertex(Int_t i, Float_t x, Float_t y, Float_t z)
{
   // Set vertex 'i'.

   fVertices[i][0] = x;
   fVertices[i][1] = y;
   fVertices[i][2] = z;
}

//______________________________________________________________________________
void TEveBox::SetVertex(Int_t i, const Float_t* v)
{
   // Set vertex 'i'.

   fVertices[i][0] = v[0];
   fVertices[i][1] = v[1];
   fVertices[i][2] = v[2];
}

//______________________________________________________________________________
void TEveBox::SetVertices(const Float_t* vs)
{
   // Set vertices.

   memcpy(fVertices, vs, sizeof(fVertices));
}

//==============================================================================

//______________________________________________________________________________
void TEveBox::ComputeBBox()
{
   // Compute bounding-box of the data.

   BBoxInit();
   for (Int_t i=0; i<8; ++i)
   {
      BBoxCheckPoint(fVertices[i]);
   }
}
