// @(#)root/g3d:$Id$
// Author: Rene Brun   03/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMaterial.h"

#include "TBuffer.h"
#include "TGeometry.h"

ClassImp(TMaterial);

/** \class TMaterial
\ingroup g3d
Manages a detector material. See class TGeometry
*/

////////////////////////////////////////////////////////////////////////////////
/// Material default constructor.

TMaterial::TMaterial()
{
   fA = 0;
   fDensity = 0;
   fInterLength = 0;
   fNumber = 0;
   fRadLength = 0;
   fZ = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Material normal constructor.

TMaterial::TMaterial(const char *name, const char *title, Float_t a, Float_t z, Float_t density)
           :TNamed(name,title), TAttFill(0,1)
{
   if (!gGeometry) gGeometry = new TGeometry("Geometry","Default Geometry");
   fA       = a;
   fZ       = z;
   fDensity = density;
   fNumber  = gGeometry->GetListOfMaterials()->GetSize();
   fRadLength   = 0;
   fInterLength = 0;
   gGeometry->GetListOfMaterials()->Add(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Material normal constructor.

TMaterial::TMaterial(const char *name, const char *title, Float_t a, Float_t z, Float_t density, Float_t radl, Float_t inter)
           :TNamed(name,title), TAttFill(0,1)
{
   if (!gGeometry) gGeometry = new TGeometry("Geometry","Default Geometry");
   fA       = a;
   fZ       = z;
   fDensity = density;
   fNumber  = gGeometry->GetListOfMaterials()->GetSize();
   fRadLength   = radl;
   fInterLength = inter;
   gGeometry->GetListOfMaterials()->Add(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Material default destructor.

TMaterial::~TMaterial()
{
   if (gGeometry) gGeometry->GetListOfMaterials()->Remove(this);

}

////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class TMaterial.

void TMaterial::Streamer(TBuffer &R__b)
{
   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      R__b.ClassBegin(TMaterial::IsA());
      R__b.ClassMember("TNamed");
      TNamed::Streamer(R__b);
      R__b.ClassMember("fNumber", "Int_t");
      R__b >> fNumber;
      R__b.ClassMember("fA", "Float_t");
      R__b >> fA;
      R__b.ClassMember("fZ", "Float_t");
      R__b >> fZ;
      R__b.ClassMember("fDensity", "Float_t");
      R__b >> fDensity;
      if (R__v > 2) {
         R__b.ClassMember("TAttFill");
         TAttFill::Streamer(R__b);
         R__b.ClassMember("fRadLength", "Float_t");
         R__b >> fRadLength;
         R__b.ClassMember("fInterLength", "Float_t");
         R__b >> fInterLength;
      } else {
         fRadLength   = 0;
         fInterLength = 0;
      }
      R__b.ClassEnd(TMaterial::IsA());
      R__b.CheckByteCount(R__s, R__c, TMaterial::IsA());
   } else {
      R__c = R__b.WriteVersion(TMaterial::IsA(), kTRUE);
      R__b.ClassBegin(TMaterial::IsA());
      R__b.ClassMember("TNamed");
      TNamed::Streamer(R__b);
      R__b.ClassMember("fNumber", "Int_t");
      R__b << fNumber;
      R__b.ClassMember("fA", "Float_t");
      R__b << fA;
      R__b.ClassMember("fZ", "Float_t");
      R__b << fZ;
      R__b.ClassMember("fDensity", "Float_t");
      R__b << fDensity;
      R__b.ClassMember("TAttFill");
      TAttFill::Streamer(R__b);
      R__b.ClassMember("fRadLength", "Float_t");
      R__b << fRadLength;
      R__b.ClassMember("fInterLength", "Float_t");
      R__b << fInterLength;
      R__b.ClassEnd(TMaterial::IsA());
      R__b.SetByteCount(R__c, kTRUE);
   }
}
