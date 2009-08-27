// @(#)root/gviz3d:$Id$
// Author: Tomasz Sosnicki   18/09/09

/************************************************************************
* Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/


#include "TStructNodeEditor.h"
#include "TStructNode.h"
#include "TStructNodeProperty.h"

#include <TGColorSelect.h>
#include <TColor.h>
#include <TGNumberEntry.h>
#include <TGLabel.h>
#include <TGTextEntry.h>
#include <TClass.h>

ClassImp(TStructNodeEditor)

//________________________________________________________________________
//////////////////////////////////////////////////////////////////////////
// 
// TStructNodeEditor is an editor for changing node attributes such as 
// maximum numbers of level or maximum number of objects diplayed if this 
// node is our top node. We can also change color associated with a class
// or chagne color to default. 
// 
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TStructNodeEditor::TStructNodeEditor(TList* colors, const TGWindow *p, Int_t width, Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, width, height, options | kVerticalFrame, back), fColors(colors)
{
   // Constructor of node attributes GUI.

   MakeTitle("TStructNode");
   fAvoidSignal = kTRUE;

   fTypeName = new TGLabel(this);
   TGLayoutHints* expandX = new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 5,5,5,5);
   this->AddFrame(fTypeName, expandX);

   TGHorizontalFrame* maxObjectsFrame = new TGHorizontalFrame(this);
   TGLabel* fMaxObjectslabel = new TGLabel(maxObjectsFrame, "Max objects:");
   maxObjectsFrame->AddFrame(fMaxObjectslabel);

   fMaxObjectsNumberEntry = new TGNumberEntry(maxObjectsFrame, 0);
   fMaxObjectsNumberEntry->SetFormat(TGNumberEntry::kNESInteger);
   maxObjectsFrame->AddFrame(fMaxObjectsNumberEntry);
   this->AddFrame(maxObjectsFrame, expandX);

   TGHorizontalFrame* maxLevelFrame = new TGHorizontalFrame(this);
   TGLabel* fMaxLevelsLabel = new TGLabel(maxLevelFrame, "Max levels:");
   maxLevelFrame->AddFrame(fMaxLevelsLabel);
   fMaxLevelsNumberEntry = new TGNumberEntry(maxLevelFrame, 0);
   fMaxLevelsNumberEntry->SetFormat(TGNumberEntry::kNESInteger);
   maxLevelFrame->AddFrame(fMaxLevelsNumberEntry);
   this->AddFrame(maxLevelFrame, expandX);

   fNameEntry = new TGTextEntry(this, fName.Data());
   this->AddFrame(fNameEntry, expandX);

   fColorSelect = new TGColorSelect(this);
   fColorSelect->Connect("ColorSelected(Pixel_t)", "TStructNodeEditor", this, "ColorSelectedSlot(Pixel_t)");
   this->AddFrame(fColorSelect, expandX);

   TGTextButton* defaultButton = new TGTextButton(this, "Default color");
   defaultButton->Connect("Clicked()", "TStructNodeEditor", this, "DefaultButtonSlot()");
   this->AddFrame(defaultButton, expandX);

   TGTextButton* applyButton = new TGTextButton(this, "Apply");
   applyButton->Connect("Clicked()", "TStructNodeEditor", this, "ApplyButtonSlot()");
   this->AddFrame(applyButton, expandX);
}

//______________________________________________________________________________
TStructNodeEditor::~TStructNodeEditor()
{ 
   // Destructor of node editor.
}

//________________________________________________________________________
void TStructNodeEditor::ApplyButtonSlot()
{
   // ApplyButton Slot. Activated when user press Apply button. Sets properties of a node

   Bool_t needReset = false;

   if ((Int_t)(fNode->GetMaxLevel()) != fMaxLevelsNumberEntry->GetIntNumber()) {
      fNode->SetMaxLevel(fMaxLevelsNumberEntry->GetIntNumber());
      needReset = true;
   }

   if ((Int_t)(fNode->GetMaxObjects()) != fMaxObjectsNumberEntry->GetIntNumber()) {
      fNode->SetMaxObjects(fMaxObjectsNumberEntry->GetIntNumber());
      needReset = true;
   }

   if (fSelectedPropert) {
      fSelectedPropert->SetColor(fColorSelect->GetColor());
      fSelectedPropert->SetName(fNameEntry->GetText());
   }

   Update(needReset);
}

//______________________________________________________________________________
void TStructNodeEditor::ColorSelectedSlot(Pixel_t color)
{
   // Slot connected to the fill area color.

   if (fAvoidSignal) {
      return;
   }

   TStructNodeProperty* prop = FindNodeProperty(fNode);
   if (prop) {
      prop->SetColor(color);
   } else {
      // add property
      prop = new TStructNodeProperty(fNode->GetTypeName(), color);
      fColors->Add(prop);
      fColors->Sort();
      fSelectedPropert = prop;
      fNameEntry->SetText(fNode->GetTypeName());
   }
   Update();
}

//________________________________________________________________________
void TStructNodeEditor::DefaultButtonSlot()
{
   // Slot for Defaulf button. Sets color of class to default

   if (TStructNodeProperty* prop = FindNodeProperty(fNode)) {
      fColors->Remove(prop);
      fSelectedPropert = GetDefaultProperty();
      fNameEntry->SetText(fSelectedPropert->GetName());
      fColorSelect->SetColor(fSelectedPropert->GetPixel(), kFALSE);
      Update();
   }
}

//________________________________________________________________________
TStructNodeProperty* TStructNodeEditor::GetDefaultProperty()
{
   // Returns property with default color 

   return (TStructNodeProperty*)fColors->Last();
}

//________________________________________________________________________
TStructNodeProperty* TStructNodeEditor::FindNodeProperty(TStructNode* node)
{
   // Retruns property associated to the class of given node "node". If property isn't found
   // then returns NULL
      
   TIter it(fColors);
   TStructNodeProperty* prop;
   while ((prop = (TStructNodeProperty*) it() )) {
      TString propName(prop->GetName());
      if (propName.EndsWith("+")) {

         if (TClass* cl = TClass::GetClass(node->GetTypeName())) {
            propName.Remove(propName.Length()-1, 1);
            if (cl->InheritsFrom(propName.Data())) {
               return prop;
            }
         }
      } else {
         if (propName == TString(node->GetTypeName())) {
            return prop;
         }
      }
   }

   return NULL;
}

//______________________________________________________________________________
void TStructNodeEditor::SetModel(TObject* obj)
{
   // Pick up the used node attributes.

   fNode = dynamic_cast<TStructNode *>(obj);

   // Add max level
   fMaxLevelsNumberEntry->SetIntNumber(fNode->GetMaxLevel());

   // Add max objects
   fMaxObjectsNumberEntry->SetIntNumber(fNode->GetMaxObjects());

   // Type label
   fTypeName->SetText(fNode->GetTypeName());

   // Add color property
   fSelectedPropert = FindNodeProperty(fNode);
   if (!fSelectedPropert)
   {
      fSelectedPropert = GetDefaultProperty();
   }
   fNameEntry->SetText(fSelectedPropert->GetName());
   fColorSelect->SetColor(fSelectedPropert->GetPixel(), kFALSE);

   fAvoidSignal = kFALSE;
}

//________________________________________________________________________
void TStructNodeEditor::Update()
{
   // Signal emmited when color or other property like number of level is changed
   // without camera reset

   Emit("Update(Bool_t)", false);
}

//________________________________________________________________________
void TStructNodeEditor::Update(Bool_t resetCamera)
{
   // Signal emmited when color or other property like number of level is changed.
   // If "resetCamera" is true, then current camera is reset.

   Emit("Update(Bool_t)", resetCamera);
}
