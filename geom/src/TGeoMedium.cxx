// @(#)root/geom:$Name:  $:$Id: TGeoMedium.cxx,v 1.5 2002/12/03 16:01:39 brun Exp $
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
