// @(#)root/geom:$Name:  $:$Id: TGeoMedium.cxx,v 1.1 2003/01/06 17:06:25 brun Exp $
// Author: Rene Brun   26/12/02

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
#include "Riostream.h"
#include "TGeoManager.h"
#include "TGeoMedium.h"

ClassImp(TGeoMedium)

//-----------------------------------------------------------------------------
TGeoMedium::TGeoMedium()
{
// Default constructor
   fId      = 0;
   fMaterial= 0;
}

//-----------------------------------------------------------------------------
TGeoMedium::TGeoMedium(const char *name, Int_t numed, const TGeoMaterial *mat, Double_t *params)
             :TNamed(name,"")
{
// constructor
   fId    = numed;
   fMaterial = (TGeoMaterial*)mat;
   for (Int_t i=0;i<10;i++) {
      if (params) fParams[i] = params[i];
      else        fParams[i] = 0;
   }
   gGeoManager->GetListOfMedia()->Add(this);
}

//-----------------------------------------------------------------------------
TGeoMedium::TGeoMedium(const char *name, Int_t numed, Int_t imat, Int_t isvol, Int_t ifield,
              Double_t fieldm, Double_t tmaxfd, Double_t stemax, Double_t deemax, Double_t epsil, Double_t stmin)
             :TNamed(name,"")
{
// constructor
   fId    = numed;
   TIter next (gGeoManager->GetListOfMaterials());
   TGeoMaterial *mat;
   while ((mat = (TGeoMaterial*)next())) {
     if (mat->GetUniqueID() == (UInt_t)imat) break;
   }
   if (mat->GetUniqueID() != (UInt_t)imat) {
      fMaterial = 0;
      Error("TGeoMedium: %s, material number %d does not exist",name,imat);
      return;
   }
   fMaterial = (TGeoMaterial*)mat;
   fParams[0] = isvol;
   fParams[1] = ifield;
   fParams[2] = fieldm;
   fParams[3] = tmaxfd;
   fParams[4] = stemax;
   fParams[5] = deemax;
   fParams[6] = epsil;
   fParams[7] = stmin;
   for (Int_t i=8;i<20;i++) fParams[i] = 0;
   gGeoManager->GetListOfMedia()->Add(this);
}

//-----------------------------------------------------------------------------
TGeoMedium::~TGeoMedium()
{
// Destructor
}

//_____________________________________________________________________________
void TGeoMedium::SavePrimitive(ofstream &out, Option_t *option)
{
// Save a primitive as a C++ statement(s) on output stream "out".
   if (TestBit(TGeoMedium::kMedSavePrimitive)) return;
   Bool_t ispmat = kTRUE;
   if (fMaterial->TestBit(TGeoMaterial::kMatSavePrimitive)) {
      out << "   // retrieve material #" << fMaterial->GetIndex() << endl;
      out << "   idmat = " << fMaterial->GetIndex() << "; // material id" <<  endl;
      out << "   pMat = gGeoManager->GetMaterial(idmat);" << endl;
   } else {
      fMaterial->SavePrimitive(out,option);
      if (fMaterial->IsMixture()) ispmat=kFALSE;
   }      
   out << "// Medium: " << GetName() << endl;
   out << "   numed   = " << fId << ";  // medium number" << endl;
   out << "   par[0]  = " << fParams[0] << "; // isvol" << endl;
   out << "   par[1]  = " << fParams[1] << "; // ifield" << endl;
   out << "   par[2]  = " << fParams[2] << "; // fieldm" << endl;
   out << "   par[3]  = " << fParams[3] << "; // tmaxfd" << endl;
   out << "   par[4]  = " << fParams[4] << "; // stemax" << endl;
   out << "   par[5]  = " << fParams[5] << "; // deemax" << endl;
   out << "   par[6]  = " << fParams[6] << "; // epsil" << endl;
   out << "   par[7]  = " << fParams[7] << "; // stmin" << endl;
   
   out << "   pMed = new TGeoMedium(\"" << GetName() << "\", numed,";
   if (ispmat) out << " pMat, par);" << endl;
   else         out << " pMix, par);" << endl;
   SetBit(TGeoMedium::kMedSavePrimitive);
}
