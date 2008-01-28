// @(#)root/eve:$Id$
// Authors: Alja & Matevz Tadel 2008

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveText.h"

#include "TFTGLManager.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TString.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TVirtualPad.h"
#include "TVirtualViewer3D.h"
#include "TMath.h"

//______________________________________________________________________________
// TEveText
//
// TEveElement class used for displaying FreeType GL fonts. Holds a
// set of parameters to define FTGL font and its rendering style.
//

ClassImp(TEveText);

//______________________________________________________________________________
TEveText::TEveText(const Text_t* txt) :
   TEveElement(fTextColor),
   TNamed("TEveText", ""),
   TAtt3D(),
   TAttBBox(),
   fText(txt),
   fTextColor(0),

   fSize(12),
   fFile(4),
   fMode(-1),
   fExtrude(1.0f),

   fAutoBehave(kTRUE),
   fLighting(kFALSE),
   fHMTrans()
{
   // Constructor.

   SetFontMode(TFTGLManager::kPixmap);
}

/******************************************************************************/

//______________________________________________________________________________
const TGPicture* TEveText::GetListTreeIcon() 
{ 
   //return pointset icon.

   return TEveElement::fgListTreeIcons[5]; 
}

//______________________________________________________________________________
void TEveText::SetFontSize(Int_t val, Bool_t validate)
{
   // Set valid font size.

   if (validate) {
      Int_t* fsp = &TFTGLManager::GetFontSizeArray()->front();
      Int_t  ns  = TFTGLManager::GetFontSizeArray()->size();
      Int_t  idx = TMath::BinarySearch(ns, fsp, val);
      fSize = fsp[idx];
   } else {
      fSize = val;
   }
}

//______________________________________________________________________________
void TEveText::SetFontFile(const char* name)
{
   // Set font file regarding to staticTFTGLManager fgFontFileArray.

   TObjArray* fa =TFTGLManager::GetFontFileArray();
   TIter  next_base(fa);
   TObjString* os;
   Int_t idx = 0;
   while ((os = (TObjString*) next_base()) != 0) {
      if (os->GetString() == name) {
         SetFontFile(idx);
         return;
      }
      idx++;
   }
}

//______________________________________________________________________________
void TEveText::SetFontMode( Int_t mode)
{
   // Set current font attributes.

   fMode = mode;

   Bool_t edit = (fMode > TFTGLManager::kPixmap);
   fHMTrans.SetEditRotation(edit);
   fHMTrans.SetEditScale(edit);
}


//______________________________________________________________________________
void TEveText::Paint(Option_t* )
{
   // Paint this object. Only direct rendering is supported.

   static const TEveException eH("TEveText::Paint ");

   TBuffer3D buff(TBuffer3DTypes::kGeneric);

   // Section kCore
   buff.fID           = this;
   buff.fColor        = GetMainColor();
   buff.fTransparency = GetMainTransparency();
   if (PtrMainHMTrans())
      PtrMainHMTrans()->SetBuffer3D(buff);
   buff.SetSectionsValid(TBuffer3D::kCore);

   Int_t reqSections = gPad->GetViewer3D()->AddObject(buff);
   if (reqSections != TBuffer3D::kNone)
      Error(eH, "only direct GL rendering supported.");
}

//______________________________________________________________________________
void TEveText::ComputeBBox()
{
   // Fill bounding-box information. Virtual from TAttBBox.
   // If member 'TEveFrameBox* fFrame' is set, frame's corners are
   // used as bbox.

   BBoxZero();
}
