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
#include "TEveTrans.h"

#include "TGLFontManager.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TString.h"
#include "TMath.h"

/** \class TEveText
\ingroup TEve
TEveElement class used for displaying FreeType GL fonts. Holds a
set of parameters to define FTGL font and its rendering style.
*/

ClassImp(TEveText);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveText::TEveText(const char* txt) :
   TEveElement(fTextColor),
   TNamed("TEveText", ""),
   TAtt3D(),
   TAttBBox(),
   fText(txt),
   fTextColor(0),

   fFontSize(12),
   fFontFile(4),
   fFontMode(-1),
   fExtrude(1.0f),

   fAutoLighting(kTRUE),
   fLighting(kFALSE)
{
   fPolygonOffset[0] = 0;
   fPolygonOffset[1] = 0;

   fCanEditMainColor        = kTRUE;
   fCanEditMainTransparency = kTRUE;
   InitMainTrans();
   SetFontMode(TGLFont::kPixmap);
}

////////////////////////////////////////////////////////////////////////////////
/// Set valid font size.

void TEveText::SetFontSize(Int_t val, Bool_t validate)
{
   if (validate) {
      Int_t* fsp = &TGLFontManager::GetFontSizeArray()->front();
      Int_t  ns  = TGLFontManager::GetFontSizeArray()->size();
      Int_t  idx = TMath::BinarySearch(ns, fsp, val);
      fFontSize = fsp[idx];
   } else {
      fFontSize = val;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set font file regarding to static TGLFontManager fgFontFileArray.

void TEveText::SetFontFile(const char* name)
{
   TObjArray* fa =TGLFontManager::GetFontFileArray();
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

////////////////////////////////////////////////////////////////////////////////
/// Set FTFont class ID.

void TEveText::SetFontMode( Int_t mode)
{
   fFontMode = mode;

   Bool_t edit = (fFontMode > TGLFont::kPixmap);
   TEveTrans& t = RefMainTrans();
   t.SetEditRotation(edit);
   t.SetEditScale(edit);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the scale and units used to calculate depth values.
/// See glPolygonOffset manual page.

void TEveText::SetPolygonOffset(Float_t factor, Float_t units)
{
   fPolygonOffset[0] = factor;
   fPolygonOffset[1] = units;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint this object. Only direct rendering is supported.

void TEveText::Paint(Option_t*)
{
   PaintStandard(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Fill bounding-box information. Virtual from TAttBBox.
/// If member 'TEveFrameBox* fFrame' is set, frame's corners are
/// used as bbox.

void TEveText::ComputeBBox()
{
   BBoxZero();
}

////////////////////////////////////////////////////////////////////////////////
/// Return TEveText icon.

const TGPicture* TEveText::GetListTreeIcon(Bool_t)
{
   return TEveElement::fgListTreeIcons[5];
}
