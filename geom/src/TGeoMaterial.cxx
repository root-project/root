// @(#)root/geom:$Name:  $:$Id: TGeoMaterial.cxx,v 1.8 2003/07/31 20:19:32 brun Exp $
// Author: Andrei Gheata   25/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
#include "TMath.h"

// statics and globals

ClassImp(TGeoMaterial)

//-----------------------------------------------------------------------------
TGeoMaterial::TGeoMaterial()
{
// Default constructor
   fIndex   = -1;
   fShader  = 0;
   fA       = 0;
   fZ       = 0;
   fDensity = 0;
   fRadLen  = 0;
   fIntLen  = 0;       
}
//-----------------------------------------------------------------------------
TGeoMaterial::TGeoMaterial(const char *name)
             :TNamed(name, "")
{
// constructor
   fIndex   = -1;
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
TGeoMaterial::TGeoMaterial(const char *name, Double_t a, Double_t z, 
                Double_t rho, Double_t radlen, Double_t intlen)
             :TNamed(name, "")
{
// constructor
   fShader  = 0;
   fIndex   = -1;
   fA       = a;
   fZ       = z;
   fDensity = rho;
   fRadLen  = radlen;
   if (a > 0 && fRadLen <= 0) {
      //taken grom Geant3 routine GSMATE
      const Double_t ALR2AV=1.39621E-03, AL183=5.20948;
      fRadLen = a/(ALR2AV*rho*z*(z +TGeoMaterial::ScreenFactor(z))*
             (AL183-TMath::Log(z)/3-TGeoMaterial::Coulomb(z)));
   }
   
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
Double_t TGeoMaterial::Coulomb(Double_t z)
{
   // static function
   //  Compute Coulomb correction for pair production and Brem 
   //  REFERENCE : EGS MANUAL SLAC 210 - UC32 - JUNE 78
   //                        FORMULA 2.7.17
   
   const Double_t ALPHA = 7.29927E-03;

   Double_t AZ    = ALPHA*z;
   Double_t AZ2   = AZ*AZ;
   Double_t AZ4   =   AZ2 * AZ2;
   Double_t FP    = ( 0.0083*AZ4 + 0.20206 + 1./(1.+AZ2) ) * AZ2;
   Double_t FM    = ( 0.0020*AZ4 + 0.0369  ) * AZ4;
   return FP - FM;
}


//-----------------------------------------------------------------------------
Bool_t TGeoMaterial::IsEq(const TGeoMaterial *other) const
{
// return true if the other material has the same physical properties
   if (other==this) return kTRUE;
   if (other->IsMixture()) return kFALSE;
   if (TMath::Abs(fA-other->GetA())>1E-3) return kFALSE;
   if (TMath::Abs(fZ-other->GetZ())>1E-3) return kFALSE;
   if (TMath::Abs(fDensity-other->GetDensity())>1E-6) return kFALSE;
//   if (fRadLen != other->GetRadLen()) return kFALSE;
//   if (fIntLen != other->GetIntLen()) return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
void TGeoMaterial::Print(const Option_t * /*option*/) const
{
// print characteristics of this material
   printf("index=%i: %s   %s   A=%g Z=%g rho=%g\n", fIndex, GetName(), GetTitle(),
          fA,fZ,fDensity);
}
//-----------------------------------------------------------------------------
Int_t TGeoMaterial::GetDefaultColor() const
{
   Int_t id = 1+ gGeoManager->GetListOfMaterials()->IndexOf(this);
   return (2+id%6);
}

//-----------------------------------------------------------------------------
Int_t TGeoMaterial::GetIndex()
{
// Retreive material index in the list of materials
   if (fIndex>=0) return fIndex;
   TList *matlist = gGeoManager->GetListOfMaterials();
   fIndex = matlist->IndexOf(this);
   return fIndex;
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
TGeoMixture::TGeoMixture(const char *name, Int_t nel, Double_t rho)
            :TGeoMaterial(name)
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
   for (Int_t j=0;j<fNelements;j++) {
      fZmixture[j] = 0;
      fAmixture[j] = 0;
      fWeights[j]  = 0;
   }
   fDensity = rho; //TO BE CORRECTED
   if (fDensity < 0) fDensity = 0.001;
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
   
   //compute equivalent radiation length (taken from Geant3/GSMIXT)
   const Double_t ALR2AV = 1.39621E-03 , AL183 =5.20948;
   Double_t radinv = 0;
   fA = 0;
   fZ = 0;
   for (Int_t j=0;j<fNelements;j++) {
      if (fWeights[j] <= 0) continue;
      fA += fWeights[j]*fAmixture[j];
      fZ += fWeights[j]*fZmixture[j];
      Double_t zc = fZmixture[j];
      Double_t alz = TMath::Log(zc)/3.;
      Double_t xinv = zc*(zc+TGeoMaterial::ScreenFactor(zc))*
         (AL183-alz-TGeoMaterial::Coulomb(zc))/fAmixture[j];
      radinv += xinv*fWeights[j];
   }
   radinv *= ALR2AV*fDensity;
   if (radinv > 0) fRadLen = 1/radinv;
}

//-----------------------------------------------------------------------------
Bool_t TGeoMixture::IsEq(const TGeoMaterial *other) const
{
// return true if the other material has the same physical properties
   if (other->IsEqual(this)) return kTRUE;
   if (!other->IsMixture()) return kFALSE;
   TGeoMixture *mix = (TGeoMixture*)other;
   if (!mix) return kFALSE;
   if (fNelements != mix->GetNelements()) return kFALSE;
   if (TMath::Abs(fA-other->GetA())>1E-3) return kFALSE;
   if (TMath::Abs(fZ-other->GetZ())>1E-3) return kFALSE;
   if (TMath::Abs(fDensity-other->GetDensity())>1E-6) return kFALSE;
//   if (fRadLen != other->GetRadLen()) return kFALSE;
//   if (fIntLen != other->GetIntLen()) return kFALSE;
   for (Int_t i=0; i<fNelements; i++) {
      if (TMath::Abs(fZmixture[i]-(mix->GetZmixt())[i])>1E-3) return kFALSE;
      if (TMath::Abs(fAmixture[i]-(mix->GetAmixt())[i])>1E-3) return kFALSE;
      if (TMath::Abs(fWeights[i]-(mix->GetWmixt())[i])>1E-3) return kFALSE;
   }
   return kTRUE;
}
//-----------------------------------------------------------------------------
void TGeoMixture::Print(const Option_t * /*option*/) const
{
// print characteristics of this material
   printf("%s   %s\n", GetName(), GetTitle());
   for (Int_t i=0; i<fNelements; i++) {
      printf("   Element #%i : Z=%6.2f A=%6.2f w=%6.2f\n", i, fZmixture[i],
             fAmixture[i], fWeights[i]);
   }
}


//-----------------------------------------------------------------------------
Double_t TGeoMaterial::ScreenFactor(Double_t z)
{
   // static function
   //  Compute screening factor for pair production and Bremstrahlung
   //  REFERENCE : EGS MANUAL SLAC 210 - UC32 - JUNE 78
   //                        FORMULA 2.7.22
   
   const Double_t AL183= 5.20948 , AL1440 = 7.27239;
   Double_t ALZ  = TMath::Log(z)/3.;
   Double_t factor = (AL1440 - 2*ALZ) / (AL183 - ALZ - TGeoMaterial::Coulomb(z));
   return factor;
}


