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

/** \class TEveGeoShapeExtract
\ingroup TEve
Globally positioned TGeoShape with rendering attributes and an
optional list of daughter shape-extracts.

Vessel to carry hand-picked geometry from gled to reve.
This class exists in both frameworks.
*/

ClassImp(TEveGeoShapeExtract);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveGeoShapeExtract::TEveGeoShapeExtract(const char* n, const char* t) :
   TNamed       (n,t),
   fRnrSelf     (kTRUE),
   fRnrElements (kTRUE),
   fRnrFrame    (kTRUE),
   fMiniFrame   (kTRUE),
   fShape       (0),
   fElements    (0)
{
   memset(fTrans, 0, sizeof(fTrans));
   fTrans[0] = fTrans[5] = fTrans[10] = fTrans[15] = 1;
   fRGBA [0] = fRGBA [1] = fRGBA [2]  = fRGBA [3]  = 1;
   fRGBALine[0] = fRGBALine[1] = fRGBALine[2] = 0; fRGBALine[3] = 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor. Delete shape and elements.

TEveGeoShapeExtract::~TEveGeoShapeExtract()
{
   delete fShape;
   delete fElements;
}

////////////////////////////////////////////////////////////////////////////////
/// True if has at least one element.

Bool_t TEveGeoShapeExtract::HasElements()
{
   return fElements != 0 && fElements->GetSize() > 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a child element.

void TEveGeoShapeExtract::AddElement(TEveGeoShapeExtract* gse)
{
   if (fElements == 0)
      fElements = new TList;

   fElements->Add(gse);
}

////////////////////////////////////////////////////////////////////////////////
/// Set transformation matrix.

void TEveGeoShapeExtract::SetTrans(const Double_t arr[16])
{
   for(Int_t i=0; i<16; ++i)
      fTrans[i] = arr[i];
}

////////////////////////////////////////////////////////////////////////////////
/// Set RGBA color.

void TEveGeoShapeExtract::SetRGBA(const Float_t  arr[4])
{
   for(Int_t i=0; i<4; ++i)
      fRGBA[i] = arr[i];
}

////////////////////////////////////////////////////////////////////////////////
/// Set RGBA color for line.

void TEveGeoShapeExtract::SetRGBALine(const Float_t  arr[4])
{
   for(Int_t i=0; i<4; ++i)
      fRGBALine[i] = arr[i];
}
