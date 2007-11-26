// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveGeoShapeExtract
#define ROOT_TEveGeoShapeExtract

#include <TNamed.h>

class TList;
class TGeoShape;

class TEveGeoShapeExtract : public TNamed
{
   friend class ZGeoRepacker;

   TEveGeoShapeExtract(const TEveGeoShapeExtract&);            // Not implemented
   TEveGeoShapeExtract& operator=(const TEveGeoShapeExtract&); // Not implemented

protected:
   Double_t    mTrans[16];
   Float_t     mRGBA[4];
   Bool_t      mRnrSelf;
   Bool_t      mRnrElements;
   TGeoShape*  mShape;
   TList*      mElements;

public:
   TEveGeoShapeExtract(const Text_t* n="TEveGeoShapeExtract", const Text_t* t=0);
   ~TEveGeoShapeExtract();

   Bool_t HasElements();
   void   AddElement(TEveGeoShapeExtract* gse);

   void SetTrans(const Double_t arr[16]);
   void SetRGBA (const Float_t  arr[4]);
   void SetRnrSelf(Bool_t r)     { mRnrSelf = r;     }
   void SetRnrElements(Bool_t r) { mRnrElements = r; }
   void SetShape(TGeoShape* s)   { mShape = s;       }
   void SetElements(TList* e)    { mElements = e;    }

   Double_t*  GetTrans()       { return mTrans; }
   Float_t*   GetRGBA()        { return mRGBA;  }
   Bool_t     GetRnrSelf()     { return mRnrSelf;     }
   Bool_t     GetRnrElements() { return mRnrElements; }
   TGeoShape* GetShape()       { return mShape;    }
   TList*     GetElements()    { return mElements; }

   ClassDef(TEveGeoShapeExtract, 1); // Globally positioned TGeoShape with rendering attributes and an optional list of daughter shape-extracts.
};

#endif
