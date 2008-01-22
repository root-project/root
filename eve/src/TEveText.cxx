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

#include "FTFont.h"

#include "TFTGLManager.h"
#include "TString.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TVirtualPad.h"
#include "TVirtualViewer3D.h"

//______________________________________________________________________________
// TEveText
//
// TEveElement class used for displaying FreeType GL fonts. Holds a
// set of parameters to define FTGL font and its rendering style.
//

ClassImp(TEveText);

//______________________________________________________________________________
TEveText::TEveText(const Text_t* text) :
   TEveElement(fTextColor),
   TNamed("TEveText","TEveText"),
   TAtt3D(),
   TAttBBox(),
   fText(text),
   fTextColor(0),

   fSize(12),
   fFile(4),
   fMode(TFTGLManager::kPixmap),

   fLighting(kFALSE),
   fExtrude(1.0f)
{
   // Constructor.
}

//______________________________________________________________________________
void TEveText::SetFont(Int_t size, Int_t file, Int_t mode)
{
   // Set current font attributes.

   fSize = size;
   fFile = file;
   if (fMode != mode)
   {
      fMode = mode;
      if (fMode == TFTGLManager::kBitmap || fMode == TFTGLManager::kPixmap) {
         fHMTrans.SetEditRotation(kFALSE);
         fHMTrans.SetEditScale(kFALSE);
      } else {
         fHMTrans.SetEditRotation(kTRUE);
         fHMTrans.SetEditScale(kTRUE);
      }
   }
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
