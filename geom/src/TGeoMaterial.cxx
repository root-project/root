/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata - date

////////////////////////////////////////////////////////////////////////////////
// Full description with examples and pictures
//
//
//
//
//Begin_Html
/*
<img src="gif/t_material.jpg">
*/
//End_Html
#include "TObjArray.h"
#include "TStyle.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"

// statics and globals

ClassImp(TGeoMaterial)

//-----------------------------------------------------------------------------
TGeoMaterial::TGeoMaterial()
{
// Default constructor
   fId = 0;
   fShader  = 0;
   fA       = 0;
   fZ       = 0;
   fDensity = 0;
   fRadLen  = 0;
   fIntLen  = 0;       
}
//-----------------------------------------------------------------------------
TGeoMaterial::TGeoMaterial(const char *name, const char *title)
             :TNamed(name, title)
{
// constructor
   fId = 0;
   fShader  = 0;
   fA       = 0;
   fZ       = 0;
   fDensity = 0;
   fRadLen  = 0;
   fIntLen  = 0;       
   if (!gGeoManager) {
      gGeoManager = new TGeoManager("Geometry", "default geometry");
   }
   gGeoManager->AddMaterial(this);
}
//-----------------------------------------------------------------------------
TGeoMaterial::TGeoMaterial(const char *name, const char *title, Double_t a, Double_t z, 
                Double_t rho, Double_t radlen, Double_t intlen)
             :TNamed(name, title)
{
// constructor
   fId = 0;
   fShader  = 0;
   fA       = a;
   fZ       = z;
   fDensity = rho;
   fRadLen  = radlen;
   fIntLen  = intlen;       
   if (!gGeoManager) {
      gGeoManager = new TGeoManager("Geometry", "default geometry");
   }
   gGeoManager->AddMaterial(this);
}
//-----------------------------------------------------------------------------
TGeoMaterial::~TGeoMaterial()
{
// Destructor
}
//-----------------------------------------------------------------------------
Bool_t TGeoMaterial::IsEq(TGeoMaterial *other)
{
// return true if the other material has the same physical properties
   if (fA != other->GetA()) return kFALSE;
   if (fZ != other->GetZ()) return kFALSE;
   if (fDensity != other->GetDensity()) return kFALSE;
   if (fRadLen != other->GetRadLen()) return kFALSE;
   if (fIntLen != other->GetIntLen()) return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
void TGeoMaterial::Print(const Option_t *option) const
{
// print characteristics of this material
   printf("%s   %s   Media=%d A=%6.2f Z=%6.0f rho=%6.2f\n", GetName(), GetTitle(),
          fId,fA,fZ,fDensity);
}
//-----------------------------------------------------------------------------
Int_t TGeoMaterial::GetDefaultColor()
{
   if (!fId) return gStyle->GetLineColor();
   return (2+fId%6);
}
/*************************************************************************
 * TGeoMixture - mixtures of elements 
 *
 *************************************************************************/
ClassImp(TGeoMixture)

//-----------------------------------------------------------------------------
TGeoMixture::TGeoMixture()
{
// Default constructor
   fNelements  = 0;
   fZmixture   = 0;
   fAmixture   = 0;
   fWeights    = 0;
}
//-----------------------------------------------------------------------------
TGeoMixture::TGeoMixture(const char *name, const char *title, Int_t nel)
            :TGeoMaterial(name, title)
{
// constructor
   if (nel == 0) {
      fZmixture   = 0;
      fAmixture   = 0;
      fWeights    = 0;
   } else {
      fZmixture = new Double_t[nel];
      fAmixture = new Double_t[nel];
      fWeights  = new Double_t[nel];
   }
   fNelements  = nel;
}
//-----------------------------------------------------------------------------
TGeoMixture::~TGeoMixture()
{
// Destructor
   if (fZmixture) delete[] fZmixture;
   if (fAmixture) delete[] fAmixture;
   if (fWeights)  delete[] fWeights;
}
//-----------------------------------------------------------------------------
void TGeoMixture:: DefineElement(Int_t i, Double_t a, Double_t z, Double_t weight)
{
// add an element to the mixture
   if ((i<0) || (i>fNelements)) {
      Error("DefineElement", "wrong index");
      return;
   }
   fZmixture[i] = z;
   fAmixture[i] = a;
   fWeights[i]  = weight;
}
//-----------------------------------------------------------------------------
Bool_t TGeoMixture::IsEq(TGeoMaterial *other)
{
// return true if the other material has the same physical properties
   if (!TGeoMaterial::IsEqual(other)) return kFALSE;
   TGeoMixture *mix = 0;
   mix = (TGeoMixture*)other;
   if (!mix) return kFALSE;
   if (fNelements != mix->GetNelements()) return kFALSE;
   for (Int_t i=0; i<fNelements; i++) {
      if (fZmixture[i] != (mix->GetZmixt())[i]) return kFALSE;
      if (fAmixture[i] != (mix->GetAmixt())[i]) return kFALSE;
      if (fWeights[i] != (mix->GetWmixt())[i]) return kFALSE;
   }
   return kTRUE;
}
//-----------------------------------------------------------------------------
void TGeoMixture::Print(const Option_t *option) const
{
// print characteristics of this material
   printf("%s   %s   Media=%d\n", GetName(), GetTitle(), fId);
   for (Int_t i=0; i<fNelements; i++) {
      printf("   Element #%i : Z=%6.2f A=%6.2f w=%6.2f\n", i, fZmixture[i],
             fAmixture[i], fWeights[i]);
   }
}



