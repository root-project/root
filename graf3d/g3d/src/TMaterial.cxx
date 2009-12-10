// @(#)root/g3d:$Id$
// Author: Rene Brun   03/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGeometry.h"
#include "TMaterial.h"

ClassImp(TMaterial)

//______________________________________________________________________________
//
// Manages a detector material. See class TGeometry
//

//______________________________________________________________________________
TMaterial::TMaterial()
{
   // Material default constructor.

   fA = 0;
   fDensity = 0;
   fInterLength = 0;
   fNumber = 0;
   fRadLength = 0;
   fZ = 0;
}


//______________________________________________________________________________
TMaterial::TMaterial(const char *name, const char *title, Float_t a, Float_t z, Float_t density)
           :TNamed(name,title), TAttFill(0,1)
{
   // Material normal constructor.

   if (!gGeometry) gGeometry = new TGeometry("Geometry","Default Geometry");
   fA       = a;
   fZ       = z;
   fDensity = density;
   fNumber  = gGeometry->GetListOfMaterials()->GetSize();
   fRadLength   = 0;
   fInterLength = 0;
   gGeometry->GetListOfMaterials()->Add(this);
}


//______________________________________________________________________________
TMaterial::TMaterial(const char *name, const char *title, Float_t a, Float_t z, Float_t density, Float_t radl, Float_t inter)
           :TNamed(name,title), TAttFill(0,1)
{
   // Material normal constructor.

   if (!gGeometry) gGeometry = new TGeometry("Geometry","Default Geometry");
   fA       = a;
   fZ       = z;
   fDensity = density;
   fNumber  = gGeometry->GetListOfMaterials()->GetSize();
   fRadLength   = radl;
   fInterLength = inter;
   gGeometry->GetListOfMaterials()->Add(this);
}


//______________________________________________________________________________
TMaterial::~TMaterial()
{
   // Material default destructor.

   if (gGeometry) gGeometry->GetListOfMaterials()->Remove(this);

}


//______________________________________________________________________________
void TMaterial::Streamer(TBuffer &R__b)
{
   // Stream an object of class TMaterial.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      TNamed::Streamer(R__b);
      R__b >> fNumber;
      R__b >> fA;
      R__b >> fZ;
      R__b >> fDensity;
      if (R__v > 2) {
         TAttFill::Streamer(R__b);
         R__b >> fRadLength;
         R__b >> fInterLength;
      } else {
         fRadLength   = 0;
         fInterLength = 0;
      }
      R__b.CheckByteCount(R__s, R__c, TMaterial::IsA());
   } else {
      R__c = R__b.WriteVersion(TMaterial::IsA(), kTRUE);
      TNamed::Streamer(R__b);
      R__b << fNumber;
      R__b << fA;
      R__b << fZ;
      R__b << fDensity;
      TAttFill::Streamer(R__b);
      R__b << fRadLength;
      R__b << fInterLength;
      R__b.SetByteCount(R__c, kTRUE);
   }
}
