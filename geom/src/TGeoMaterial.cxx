// @(#)root/geom:$Name:  $:$Id: TGeoMaterial.cxx,v 1.28 2006/05/23 04:47:37 brun Exp $
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
#include "Riostream.h"
#include "TMath.h"
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
   SetUsed(kFALSE);
   fIndex    = -1;
   fShader   = 0;
   fA        = 0;
   fZ        = 0;
   fDensity  = 0;
   fRadLen   = 0;
   fIntLen   = 0;       
   fCerenkov = 0;
}
//-----------------------------------------------------------------------------
TGeoMaterial::TGeoMaterial(const char *name)
             :TNamed(name, "")
{
// constructor
   fName = fName.Strip();
   SetUsed(kFALSE);
   fIndex    = -1;
   fShader   = 0;
   fA        = 0;
   fZ        = 0;
   fDensity  = 0;
   fRadLen   = 0;
   fIntLen   = 0;
   fCerenkov = 0;
   
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
   fName = fName.Strip();
   SetUsed(kFALSE);
   fShader   = 0;
   fIndex    = -1;
   fA        = a;
   fZ        = z;
   fDensity  = rho;
   fCerenkov = 0;
   SetRadLen(radlen, intlen);
   if (!gGeoManager) {
      gGeoManager = new TGeoManager("Geometry", "default geometry");
   }
   if (fZ - Int_t(fZ) > 1E-3)
      Warning("ctor", "Material %s defined with fractional Z=%f", GetName(), fZ);
   GetElement()->SetUsed();
   gGeoManager->AddMaterial(this);
}
//-----------------------------------------------------------------------------
TGeoMaterial::TGeoMaterial(const char *name, TGeoElement *elem,
                Double_t rho)
             :TNamed(name, "")
{
// constructor
   fName = fName.Strip();
   SetUsed(kFALSE);
   fShader   = 0;
   fIndex    = -1;
   fA        = elem->A();
   fZ        = elem->Z();
   fDensity  = rho;
   fCerenkov = 0;
   
   SetRadLen(0,0);
   if (!gGeoManager) {
      gGeoManager = new TGeoManager("Geometry", "default geometry");
   }
   if (fZ - Int_t(fZ) > 1E-3)
      Warning("ctor", "Material %s defined with fractional Z=%f", GetName(), fZ);
   GetElement()->SetUsed();
   gGeoManager->AddMaterial(this);
}
//-----------------------------------------------------------------------------
TGeoMaterial::TGeoMaterial(const TGeoMaterial& gm) :
  TNamed(gm),
  TAttFill(gm),
  fIndex(gm.fIndex),
  fA(gm.fA),
  fZ(gm.fZ),
  fDensity(gm.fDensity),
  fRadLen(gm.fRadLen),
  fIntLen(gm.fIntLen),
  fShader(gm.fShader),
  fCerenkov(gm.fCerenkov)
{ 
   //copy constructor
}
//-----------------------------------------------------------------------------
TGeoMaterial& TGeoMaterial::operator=(const TGeoMaterial& gm) 
{
   //equal operator
   if(this!=&gm) {
      TNamed::operator=(gm);
      TAttFill::operator=(gm);
      fIndex=gm.fIndex;
      fA=gm.fA;
      fZ=gm.fZ;
      fDensity=gm.fDensity;
      fRadLen=gm.fRadLen;
      fIntLen=gm.fIntLen;
      fShader=gm.fShader;
      fCerenkov=gm.fCerenkov;
   } 
   return *this;
}
//-----------------------------------------------------------------------------
TGeoMaterial::~TGeoMaterial()
{
// Destructor
}

//_____________________________________________________________________________
char *TGeoMaterial::GetPointerName() const
{
// Provide a pointer name containing uid.
   static char name[20];
   sprintf(name,"pMat%d", GetUniqueID());
   return name;
}    

//-----------------------------------------------------------------------------
void TGeoMaterial::SetRadLen(Double_t radlen, Double_t intlen)
{
// Set radiation/absorbtion lengths. If the values are negative, their absolute value
// is taken, otherwise radlen is recomputed using G3 formula.
   fRadLen = TMath::Abs(radlen);
   fIntLen = TMath::Abs(intlen);
   // compute radlen systematically with G3 formula for a valid material
   if (fA > 0 && fZ > 0 && radlen>=0) {
      //taken grom Geant3 routine GSMATE
      const Double_t alr2av=1.39621E-03, al183=5.20948;
      fRadLen = fA/(alr2av*fDensity*fZ*(fZ +TGeoMaterial::ScreenFactor(fZ))*
             (al183-TMath::Log(fZ)/3-TGeoMaterial::Coulomb(fZ)));
   } else {
      if (radlen>0) Error("SetRadLen","Invalid material %s: a=%f z=%f -> user values taken: radlen=%f intlen=%f",fA,fZ,radlen,intlen);
   }   
}   

//-----------------------------------------------------------------------------
Double_t TGeoMaterial::Coulomb(Double_t z)
{
   // static function
   //  Compute Coulomb correction for pair production and Brem 
   //  REFERENCE : EGS MANUAL SLAC 210 - UC32 - JUNE 78
   //                        FORMULA 2.7.17
   
   const Double_t alpha = 7.29927E-03;

   Double_t az    = alpha*z;
   Double_t az2   = az*az;
   Double_t az4   =   az2 * az2;
   Double_t fp    = ( 0.0083*az4 + 0.20206 + 1./(1.+az2) ) * az2;
   Double_t fm    = ( 0.0020*az4 + 0.0369  ) * az4;
   return fp - fm;
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
   if (GetCerenkovProperties() != other->GetCerenkovProperties()) return kFALSE;
//   if (fRadLen != other->GetRadLen()) return kFALSE;
//   if (fIntLen != other->GetIntLen()) return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
void TGeoMaterial::Print(const Option_t * /*option*/) const
{
// print characteristics of this material
   printf("Material %s %s   A=%g Z=%g rho=%g radlen=%g index=%i\n", GetName(), GetTitle(),
          fA,fZ,fDensity, fRadLen, fIndex);
}

//_____________________________________________________________________________
void TGeoMaterial::SavePrimitive(ofstream &out, Option_t * /*option*/)
{
// Save a primitive as a C++ statement(s) on output stream "out".
   if (TestBit(TGeoMaterial::kMatSavePrimitive)) return;
   char *name = GetPointerName();
   out << "// Material: " << GetName() << endl;
   out << "   a       = " << fA << ";" << endl;
   out << "   z       = " << fZ << ";" << endl;
   out << "   density = " << fDensity << ";" << endl;
   out << "   radl    = " << fRadLen << ";" << endl;
   out << "   absl    = " << fIntLen << ";" << endl;
   
   out << "   " << name << " = new TGeoMaterial(\"" << GetName() << "\", a,z,density,radl,absl);" << endl;
   out << "   " << name << "->SetIndex(" << GetIndex() << ");" << endl;
   SetBit(TGeoMaterial::kMatSavePrimitive);
}

//-----------------------------------------------------------------------------
Int_t TGeoMaterial::GetDefaultColor() const
{
// Get some default color related to this material.
   Int_t id = 1+ gGeoManager->GetListOfMaterials()->IndexOf(this);
   return (2+id%6);
}

//-----------------------------------------------------------------------------
TGeoElement *TGeoMaterial::GetElement(Int_t) const
{
// Get a pointer to the element this material is made of.
   TGeoElementTable *table = gGeoManager->GetElementTable();
   return table->GetElement(Int_t(fZ));
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
TGeoMixture::TGeoMixture(const TGeoMixture& gm) :
  TGeoMaterial(gm),
  fNelements(gm.fNelements),
  fZmixture(gm.fZmixture),
  fAmixture(gm.fAmixture),
  fWeights(gm.fWeights)
{ }
//-----------------------------------------------------------------------------
TGeoMixture& TGeoMixture::operator=(const TGeoMixture& gm) 
{
  if(this!=&gm) {
    TGeoMaterial::operator=(gm);
    fNelements=gm.fNelements;
    fZmixture=gm.fZmixture;
    fAmixture=gm.fAmixture;
    fWeights=gm.fWeights;
  } return *this;
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
      Error("DefineElement", "wrong index iel=%i in mixture %s, max is %d", i, GetName(), fNelements);
      return;
   }
   fZmixture[i] = z;
   fAmixture[i] = a;
   fWeights[i]  = weight;
   if (z - Int_t(z) > 1E-3)
      Warning("DefineElement", "Mixture %s has element defined with fractional Z=%f", GetName(), z);
   GetElement(i)->SetDefined();
   
   //compute equivalent radiation length (taken from Geant3/GSMIXT)
   const Double_t alr2av = 1.39621E-03 , al183 =5.20948;
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
         (al183-alz-TGeoMaterial::Coulomb(zc))/fAmixture[j];
      radinv += xinv*fWeights[j];
   }
   radinv *= alr2av*fDensity;
   if (radinv > 0) fRadLen = 1/radinv;
}

//-----------------------------------------------------------------------------
void TGeoMixture:: DefineElement(Int_t i, TGeoElement *elem, Double_t weight)
{
// Define one component as being a given element with a specified proportion by weight.
   DefineElement(i, elem->A(), elem->Z(), weight);
}   

//-----------------------------------------------------------------------------
void TGeoMixture:: DefineElement(Int_t iel, Int_t z, Int_t natoms)
{
// Define the mixture element at index iel by number of atoms in the chemical formula.
   Int_t i;
   if ((iel<0) || (iel>fNelements)) {
      Error("DefineElement", "wrong index iel=%i in mixture %s, max is %d", iel, GetName(), fNelements);
      return;
   }
   TGeoElementTable *table = gGeoManager->GetElementTable();
   TGeoElement *elem = table->GetElement(z);
   if (!elem) Fatal("DefineElement", "In mixture %s, element with Z=%i not found",GetName(),z);
   fZmixture[iel] = elem->Z();
   fAmixture[iel] = elem->A();
   fWeights[iel]  = natoms;
   Double_t amol = 0.;
   for (i=0; i<fNelements; i++) {
      if (fWeights[i]<=0) return;
      amol += fAmixture[i]*fWeights[i];
   }   
   for (i=0; i<fNelements; i++) {
      fWeights[i] *= fAmixture[i]/amol;
      DefineElement(i, fAmixture[i], fZmixture[i], fWeights[i]);
   }
}          

//-----------------------------------------------------------------------------
TGeoElement *TGeoMixture::GetElement(Int_t i) const
{
// Retreive the pointer to the element corresponding to component I.
   if (i<0 || i>=fNelements) {
      Error("GetElement", "Mixture %s has only %d elements", GetName(), fNelements);
      return 0;
   }   
   TGeoElementTable *table = gGeoManager->GetElementTable();
   return table->GetElement(Int_t(fZmixture[i]));
}

//-----------------------------------------------------------------------------
Bool_t TGeoMixture::IsEq(const TGeoMaterial *other) const
{
// Return true if the other material has the same physical properties
   if (other->IsEqual(this)) return kTRUE;
   if (!other->IsMixture()) return kFALSE;
   TGeoMixture *mix = (TGeoMixture*)other;
   if (!mix) return kFALSE;
   if (fNelements != mix->GetNelements()) return kFALSE;
   if (TMath::Abs(fA-other->GetA())>1E-3) return kFALSE;
   if (TMath::Abs(fZ-other->GetZ())>1E-3) return kFALSE;
   if (TMath::Abs(fDensity-other->GetDensity())>1E-6) return kFALSE;
   if (GetCerenkovProperties() != other->GetCerenkovProperties()) return kFALSE;
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
   printf("Mixture %s %s   Aeff=%g Zeff=%g rho=%g radlen=%g index=%i\n", GetName(), GetTitle(),
          fA,fZ,fDensity, fRadLen, fIndex);
   for (Int_t i=0; i<fNelements; i++) {
      printf("   Element #%i : Z=%6.2f A=%6.2f w=%6.2f\n", i, fZmixture[i],
             fAmixture[i], fWeights[i]);
   }
}

//_____________________________________________________________________________
void TGeoMixture::SavePrimitive(ofstream &out, Option_t * /*option*/)
{
// Save a primitive as a C++ statement(s) on output stream "out".
   if (TestBit(TGeoMaterial::kMatSavePrimitive)) return;
   char *name = GetPointerName();
   out << "// Mixture: " << GetName() << endl;
   out << "   nel     = " << fNelements << ";" << endl;
   out << "   density = " << fDensity << ";" << endl;
   out << "   " << name << " = new TGeoMixture(\"" << GetName() << "\", nel,density);" << endl;
   for (Int_t i=0; i<fNelements; i++) {
      TGeoElement *el = GetElement(i);
      out << "      a = " << fAmixture[i] << ";   z = "<< fZmixture[i] << ";   w = " << fWeights[i] << ";  // " << el->GetName() << endl;
      out << "   " << name << "->DefineElement(" << i << ",a,z,w);" << endl;
   }         
   out << "   " << name << "->SetIndex(" << GetIndex() << ");" << endl;
   SetBit(TGeoMaterial::kMatSavePrimitive);
}

//-----------------------------------------------------------------------------
Double_t TGeoMaterial::ScreenFactor(Double_t z)
{
   // static function
   //  Compute screening factor for pair production and Bremstrahlung
   //  REFERENCE : EGS MANUAL SLAC 210 - UC32 - JUNE 78
   //                        FORMULA 2.7.22
   
   const Double_t al183= 5.20948 , al1440 = 7.27239;
   Double_t alz  = TMath::Log(z)/3.;
   Double_t factor = (al1440 - 2*alz) / (al183 - alz - TGeoMaterial::Coulomb(z));
   return factor;
}


