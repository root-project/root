// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveGeoShapeExtract.h"
#include "TEveGeoNode.h"
#include "TEveGeoShape.h"

#include "TList.h"
#include "TGeoManager.h"
#include "TGeoShape.h"

//==============================================================================
// TEveGeoShapeExtract
//==============================================================================

//______________________________________________________________________________
//
// Globally positioned TGeoShape with rendering attributes and an
// optional list of daughter shape-extracts.
//
// Vessel to carry hand-picked geometry from gled to reve.
// This class exists in both frameworks.

ClassImp(TEveGeoShapeExtract);

//______________________________________________________________________________
TEveGeoShapeExtract::TEveGeoShapeExtract(const char* n, const char* t) :
   TNamed       (n,t),
   fRnrSelf     (kTRUE),
   fRnrElements (kTRUE),
   fRnrFrame    (kTRUE),
   fMiniFrame   (kTRUE),
   fShape       (0),
   fElements    (0)
{
   // Constructor.

   memset(fTrans, 0, sizeof(fTrans));
   fTrans[0] = fTrans[5] = fTrans[10] = fTrans[15] = 1;
   fRGBA [0] = fRGBA [1] = fRGBA [2]  = fRGBA [3]  = 1;
   fRGBALine[0] = fRGBALine[1] = fRGBALine[2] = 0; fRGBALine[3] = 1;
}

//______________________________________________________________________________
TEveGeoShapeExtract::~TEveGeoShapeExtract()
{
   // Destructor. Delete shape and elements.

   delete fShape;
   delete fElements;
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveGeoShapeExtract::HasElements()
{
   // True if has at least one element.

   return fElements != 0 && fElements->GetSize() > 0;
}

//______________________________________________________________________________
void TEveGeoShapeExtract::AddElement(TEveGeoShapeExtract* gse)
{
   // Add a child element.

   if (fElements == 0)
      fElements = new TList;

   fElements->Add(gse);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoShapeExtract::SetTrans(const Double_t arr[16])
{
   // Set transformation matrix.

   for(Int_t i=0; i<16; ++i)
      fTrans[i] = arr[i];
}

//______________________________________________________________________________
void TEveGeoShapeExtract::SetRGBA(const Float_t  arr[4])
{
   // Set RGBA color.

   for(Int_t i=0; i<4; ++i)
      fRGBA[i] = arr[i];
}

//______________________________________________________________________________
void TEveGeoShapeExtract::SetRGBALine(const Float_t  arr[4])
{
   // Set RGBA color for line.

   for(Int_t i=0; i<4; ++i)
      fRGBALine[i] = arr[i];
}
