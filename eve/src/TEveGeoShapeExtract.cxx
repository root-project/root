// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TEveGeoShapeExtract.h>

#include <TList.h>
#include <TGeoShape.h>

//______________________________________________________________________________
// TEveGeoShapeExtract
//
// Globally positioned TGeoShape with rendering attributes and an
// optional list of daughter shape-extracts.
//
// Vessel to carry hand-picked geometry from gled to reve.
// This class exists in both frameworks.

ClassImp(TEveGeoShapeExtract)

//______________________________________________________________________________
TEveGeoShapeExtract::TEveGeoShapeExtract(const Text_t* n, const Text_t* t) :
   TNamed       (n,t),

   mRnrSelf     (true),
   mRnrElements (true),
   mShape       (0),
   mElements    (0)
{
   memset(mTrans, 0, sizeof(mTrans));
   mTrans[0] = mTrans[5] = mTrans[10] = mTrans[15] = 1;
   mRGBA [0] = mRGBA [1] = mRGBA [2]  = mRGBA [3]  = 1;
}

//______________________________________________________________________________
TEveGeoShapeExtract::~TEveGeoShapeExtract()
{
   delete mShape;
   delete mElements;
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveGeoShapeExtract::HasElements()
{
   return mElements != 0 && mElements->GetSize() > 0;
}

//______________________________________________________________________________
void TEveGeoShapeExtract::AddElement(TEveGeoShapeExtract* gse)
{
   if (mElements == 0)
      mElements = new TList;

   mElements->Add(gse);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoShapeExtract::SetTrans(const Double_t arr[16])
{
   for(Int_t i=0; i<16; ++i)
      mTrans[i] = arr[i];
}

//______________________________________________________________________________
void TEveGeoShapeExtract::SetRGBA (const Float_t  arr[4])
{
   for(Int_t i=0; i<4; ++i)
      mRGBA[i] = arr[i];
}
