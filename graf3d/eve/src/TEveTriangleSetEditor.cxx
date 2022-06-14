// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveTriangleSetEditor.h"
#include "TEveTriangleSet.h"

#include "TGWidget.h"
#include "TGLabel.h"

/** \class TEveTriangleSetEditor
\ingroup TEve
Editor for TEveTriangleSet class.
*/

ClassImp(TEveTriangleSetEditor);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveTriangleSetEditor::TEveTriangleSetEditor(const TGWindow *p, Int_t width, Int_t height,
                                             UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fM    (0),
   fInfo (0)
{
   MakeTitle("TEveTriangleSet");

   fInfo = new TGLabel(this);
   fInfo->SetTextJustify(kTextLeft);
   AddFrame(fInfo, new TGLayoutHints(kLHintsNormal|kLHintsExpandX, 8, 0, 2, 0));
}

////////////////////////////////////////////////////////////////////////////////
/// Set model object.

void TEveTriangleSetEditor::SetModel(TObject* obj)
{
   fM = dynamic_cast<TEveTriangleSet*>(obj);

   fInfo->SetText(Form("Vertices: %d, Triangles: %d", fM->GetNVerts(), fM->GetNTrings()));
}
