// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveGeoNodeEditor.h"
#include "TEveGValuators.h"

#include "TEveGeoNode.h"
#include "TGeoNode.h"

#include "TVirtualPad.h"
#include "TColor.h"

#include "TGLabel.h"
#include "TGButton.h"
#include "TGNumberEntry.h"
#include "TGColorSelect.h"
#include "TGDoubleSlider.h"

//______________________________________________________________________________
// TEveGeoNodeEditor
//
// Editor for TEveGeoNode class.

ClassImp(TEveGeoNodeEditor)

//______________________________________________________________________________
TEveGeoNodeEditor::TEveGeoNodeEditor(const TGWindow *p,
                                     Int_t width, Int_t height,
                                     UInt_t options, Pixel_t back) :
   TGedFrame(p,width, height, options | kVerticalFrame, back),

   fNodeRE (0),

   fVizNode(0),
   fVizNodeDaughters(0),
   fVizVolume(0),
   fVizVolumeDaughters(0)
{
   // Constructor.

   MakeTitle("GeoNode");

   // --- Visibility control

   fVizNode = new TGCheckButton(this, "VizNode");
   AddFrame(fVizNode, new TGLayoutHints(kLHintsTop, 3, 1, 1, 0));
   fVizNode->Connect
      ("Toggled(Bool_t)",
       "TEveGeoNodeEditor", this, "DoVizNode()");

   fVizNodeDaughters = new TGCheckButton(this, "VizNodeDaughters");
   AddFrame(fVizNodeDaughters, new TGLayoutHints(kLHintsTop, 3, 1, 1, 0));
   fVizNodeDaughters->Connect
      ("Toggled(Bool_t)",
       "TEveGeoNodeEditor", this, "DoVizNodeDaughters()");

   fVizVolume = new TGCheckButton(this, "VizVolume");
   AddFrame(fVizVolume, new TGLayoutHints(kLHintsTop, 3, 1, 1, 0));
   fVizVolume->Connect
      ("Toggled(Bool_t)",
       "TEveGeoNodeEditor", this, "DoVizVolume()");

   fVizVolumeDaughters = new TGCheckButton(this, "VizVolumeDaughters");
   AddFrame(fVizVolumeDaughters, new TGLayoutHints(kLHintsTop, 3, 1, 1, 0));
   fVizVolumeDaughters->Connect
      ("Toggled(Bool_t)",
       "TEveGeoNodeEditor", this, "DoVizVolumeDaughters()");
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoNodeEditor::SetModel(TObject* obj)
{
   // Set model object.

   fNodeRE = dynamic_cast<TEveGeoNode*>(obj);
   TGeoNode*  node = fNodeRE->fNode;
   TGeoVolume* vol = node->GetVolume();

   fVizNode->SetState(node->TGeoAtt::IsVisible() ? kButtonDown : kButtonUp);
   fVizNodeDaughters->SetState(node->TGeoAtt::IsVisDaughters() ? kButtonDown : kButtonUp);
   fVizVolume->SetState(vol->IsVisible() ? kButtonDown : kButtonUp);
   fVizVolumeDaughters->SetState(vol->IsVisDaughters() ? kButtonDown : kButtonUp);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoNodeEditor::DoVizNode()
{
   // Slot for VizNode.

   fNodeRE->SetRnrSelf(fVizNode->IsOn());
   Update();
}

//______________________________________________________________________________
void TEveGeoNodeEditor::DoVizNodeDaughters()
{
   // Slot for VizNodeDaughters.

   fNodeRE->SetRnrChildren(fVizNodeDaughters->IsOn());
   Update();
}

//______________________________________________________________________________
void TEveGeoNodeEditor::DoVizVolume()
{
   // Slot for VizVolume.

   fNodeRE->fNode->GetVolume()->SetVisibility(fVizVolume->IsOn());
   Update();
}

//______________________________________________________________________________
void TEveGeoNodeEditor::DoVizVolumeDaughters()
{
   // Slot for VizVolumeDaughters.

   fNodeRE->fNode->GetVolume()->VisibleDaughters(fVizVolumeDaughters->IsOn());
   Update();
}


//______________________________________________________________________________
// TEveGeoTopNodeEditor
//
// Editor for TEveGeoTopNode class.

ClassImp(TEveGeoTopNodeEditor)

//______________________________________________________________________________
TEveGeoTopNodeEditor::TEveGeoTopNodeEditor(const TGWindow *p,
                                           Int_t width, Int_t height,
                                           UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),

   fTopNodeRE   (0),
   fVisOption   (0),
   fVisLevel    (0),
   fMaxVisNodes (0)
{
   // Constructor.

   MakeTitle("GeoTopNode");

   Int_t labelW = 64;

   fVisOption = new TEveGValuator(this, "VisOption:", 90, 0);
   fVisOption->SetLabelWidth(labelW);
   fVisOption->SetShowSlider(kFALSE);
   fVisOption->SetNELength(4);
   fVisOption->Build();
   fVisOption->SetLimits(0, 2, 10, TGNumberFormat::kNESInteger);
   fVisOption->SetToolTip("Visualization option passed to TGeoPainter.");
   fVisOption->Connect("ValueSet(Double_t)", "TEveGeoTopNodeEditor", this, "DoVisOption()");
   AddFrame(fVisOption, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));

   fVisLevel = new TEveGValuator(this, "VisLevel:", 90, 0);
   fVisLevel->SetLabelWidth(labelW);
   fVisLevel->SetShowSlider(kFALSE);
   fVisLevel->SetNELength(4);
   fVisLevel->Build();
   fVisLevel->SetLimits(0, 30, 31, TGNumberFormat::kNESInteger);
   fVisLevel->SetToolTip("Level (depth) to which the geometry is traversed.\nWhen zero, maximum number of nodes to draw can be specified.");
   fVisLevel->Connect("ValueSet(Double_t)", "TEveGeoTopNodeEditor", this, "DoVisLevel()");
   AddFrame(fVisLevel, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));

   fMaxVisNodes = new TEveGValuator(this, "MaxNodes:", 90, 0);
   fMaxVisNodes->SetLabelWidth(labelW);
   fMaxVisNodes->SetShowSlider(kFALSE);
   fMaxVisNodes->SetNELength(6);
   fMaxVisNodes->Build();
   fMaxVisNodes->SetLimits(100, 999999, 0, TGNumberFormat::kNESInteger);
   fMaxVisNodes->SetToolTip("Maximum number of nodes to draw.");
   fMaxVisNodes->Connect("ValueSet(Double_t)", "TEveGeoTopNodeEditor", this, "DoMaxVisNodes()");
   AddFrame(fMaxVisNodes, new TGLayoutHints(kLHintsTop, 1, 1, 1, 1));
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoTopNodeEditor::SetModel(TObject* obj)
{
   // Set model object.

   fTopNodeRE = dynamic_cast<TEveGeoTopNode*>(obj);

   fVisOption  ->SetValue(fTopNodeRE->GetVisOption());
   fVisLevel   ->SetValue(fTopNodeRE->GetVisLevel());
   fMaxVisNodes->SetValue(fTopNodeRE->GetMaxVisNodes());
   if (fTopNodeRE->GetVisLevel() > 0)
      fMaxVisNodes->UnmapWindow();
   else
      fMaxVisNodes->MapWindow();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGeoTopNodeEditor::DoVisOption()
{
   // Slot for VisOption.

   fTopNodeRE->SetVisOption(Int_t(fVisOption->GetValue()));
   Update();
}

//______________________________________________________________________________
void TEveGeoTopNodeEditor::DoVisLevel()
{
   // Slot for VisLevel.

   fTopNodeRE->SetVisLevel(Int_t(fVisLevel->GetValue()));
   Update();
}

//______________________________________________________________________________
void TEveGeoTopNodeEditor::DoMaxVisNodes()
{
   // Slot for MaxVisNodes.

   fTopNodeRE->SetMaxVisNodes(Int_t(fMaxVisNodes->GetValue()));
   Update();
}
