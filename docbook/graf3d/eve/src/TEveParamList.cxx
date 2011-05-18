// @(#)root/eve:$Id$
// Author: Dmytro Kovalskyi, 28.2.2008

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveParamList.h"

// Cleanup these includes:
#include "TGLabel.h"
#include "TGButton.h"
#include "TGNumberEntry.h"
#include "TGColorSelect.h"
#include "TGDoubleSlider.h"

#include "TEveGValuators.h"
#include "TGNumberEntry.h"
#include "TGedEditor.h"

//==============================================================================
//==============================================================================
// TEveParamList
//==============================================================================

//______________________________________________________________________________
//
// Collection of named parameters.

ClassImp(TEveParamList);

//______________________________________________________________________________
TEveParamList::TEveParamList(const char* n, const char* t, Bool_t doColor) :
   TNamed(n, t),
   fColor(0)
{
   // Constructor.

   if (doColor) SetMainColorPtr(&fColor);
}

//______________________________________________________________________________
TEveParamList::FloatConfig_t TEveParamList::GetFloatParameter(const TString& name)
{
   // Get config-struct for float parameter 'name'.

   static const TEveException eh("TEveParamList::GetFloatParameter ");

   for (FloatConfigVec_ci itr = fFloatParameters.begin(); itr != fFloatParameters.end(); ++itr)
      if (itr->fName.CompareTo(name)==0 ) return *itr;
   Error(eh, "parameter not found.");
   return FloatConfig_t();
}

//______________________________________________________________________________
TEveParamList::IntConfig_t TEveParamList::GetIntParameter(const TString& name)
{
   // Get config-struct for int parameter 'name'.

   static const TEveException eh("TEveParamList::GetIntParameter ");

   for (IntConfigVec_ci itr = fIntParameters.begin(); itr != fIntParameters.end(); ++itr)
      if (itr->fName.CompareTo(name) == 0) return *itr;
   Error(eh, "parameter not found.");
   return IntConfig_t();
}

//______________________________________________________________________________
Bool_t TEveParamList::GetBoolParameter(const TString& name)
{
   // Get value for bool parameter 'name'.

   static const TEveException eh("TEveParamList::GetBoolParameter ");

   for (BoolConfigVec_ci itr = fBoolParameters.begin(); itr != fBoolParameters.end(); ++itr)
      if ( itr->fName.CompareTo(name)==0 ) return itr->fValue;
   Error(eh, "parameter not found.");
   return kFALSE;
}

//______________________________________________________________________________
void TEveParamList::ParamChanged(const char* name)
{
   // Emit ParamChanged() signal.

   Emit("ParamChanged(char*)", name);
}

//==============================================================================
//==============================================================================
// TEveParamListEditor
//==============================================================================

//______________________________________________________________________________
// GUI editor for TEveParamList.
//
// Slot methods from this object do not call Update, instead they call
// their model's ParamChanged(const char* name) function which emits a
// corresponding signal.
//
// This helps in handling of parameter changes as they are probably
// related to displayed objects in a more complicated way. Further,
// the TGCheckButton::HandleButton() emits more signal after the
// Clicked() signal and if model is reset in the editor, its contents
// is removed. This results in a crash.

ClassImp(TEveParamListEditor);

//______________________________________________________________________________
TEveParamListEditor::TEveParamListEditor(const TGWindow *p, Int_t width, Int_t height,
                                         UInt_t options, Pixel_t back) :
   TGedFrame(p, width, height, options | kVerticalFrame, back),
   fM          (0),
   fParamFrame (0)
{
   // Constructor.

   MakeTitle("TEveParamList");
}

//______________________________________________________________________________
void TEveParamListEditor::InitModel(TObject* obj)
{
   // Initialize widgets when a new object is selected.

   fM = dynamic_cast<TEveParamList*>(obj);

   if (fParamFrame) {
      fParamFrame->UnmapWindow();
      RemoveFrame(fParamFrame);
      fParamFrame->DestroyWindow();
      delete fParamFrame;
   }
   fParamFrame = new TGVerticalFrame(this);
   AddFrame(fParamFrame);

   // integers
   fIntParameters.clear();
   for (UInt_t i = 0; i < fM->fIntParameters.size(); ++i)
   {
      TGCompositeFrame* frame = new TGHorizontalFrame(fParamFrame);

      // number entry widget
      TGNumberEntry* widget = new TGNumberEntry
         (frame, fM->fIntParameters[i].fValue,
          5,                                 // number of digits
          i,                                 // widget ID
          TGNumberFormat::kNESInteger,       // style
          TGNumberFormat::kNEAAnyNumber,     // input value filter
          TGNumberFormat::kNELLimitMinMax,   // specify limits
          fM->fIntParameters[i].fMin,        // min value
          fM->fIntParameters[i].fMax);       // max value
      frame->AddFrame(widget, new TGLayoutHints(kLHintsLeft|kLHintsCenterY, 2,8,2,2));
      widget->Connect("ValueSet(Long_t)", "TEveParamListEditor", this, "DoIntUpdate()");
      fIntParameters.push_back(widget);

      // label
      frame->AddFrame(new TGLabel(frame,fM->fIntParameters[i].fName.Data()),
                      new TGLayoutHints(kLHintsLeft|kLHintsCenterY));

      fParamFrame->AddFrame(frame, new TGLayoutHints(kLHintsTop));
   }


   // floats
   fFloatParameters.clear();
   for (UInt_t i = 0; i < fM->fFloatParameters.size(); ++i)
   {
      TGCompositeFrame* frame = new TGHorizontalFrame(fParamFrame);

      // number entry widget
      TGNumberEntry* widget = new TGNumberEntry
         (frame, fM->fFloatParameters[i].fValue,
          5,                                // number of digits
          i,                                // widget ID
          TGNumberFormat::kNESReal,         // style
          TGNumberFormat::kNEAAnyNumber,    // input value filter
          TGNumberFormat::kNELLimitMinMax,  // specify limits
          fM->fFloatParameters[i].fMin,     // min value
          fM->fFloatParameters[i].fMax);    // max value
      frame->AddFrame(widget, new TGLayoutHints(kLHintsLeft|kLHintsCenterY, 2,8,2,2));
      widget->Connect("ValueSet(Long_t)", "TEveParamListEditor", this, "DoFloatUpdate()");
      fFloatParameters.push_back( widget );

      // label
      frame->AddFrame(new TGLabel(frame,fM->fFloatParameters[i].fName.Data()),
                      new TGLayoutHints(kLHintsLeft|kLHintsCenterY) );

      fParamFrame->AddFrame(frame, new TGLayoutHints(kLHintsTop));
   }

   // boolean
   fBoolParameters.clear();
   for (UInt_t i = 0; i < fM->fBoolParameters.size(); ++i)
   {
      TGCheckButton* widget = new TGCheckButton(fParamFrame,
                                                fM->fBoolParameters[i].fName.Data(),
                                                i);
      widget->Connect("Clicked()", "TEveParamListEditor", this, "DoBoolUpdate()");
      fBoolParameters.push_back(widget);

      fParamFrame->AddFrame(widget, new TGLayoutHints(kLHintsTop,2,0,1,1));
   }
   MapSubwindows();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveParamListEditor::SetModel(TObject* obj)
{
   // Set model object.

   InitModel(obj);

   for (UInt_t i = 0; i < fIntParameters.size(); ++i)
      fIntParameters[i]->GetNumberEntry()->SetIntNumber(fM->fIntParameters[i].fValue);

   for (UInt_t i = 0; i < fFloatParameters.size(); ++i)
      fFloatParameters[i]->GetNumberEntry()->SetNumber(fM->fFloatParameters[i].fValue);

   for (UInt_t i = 0; i < fBoolParameters.size(); ++i)
      fBoolParameters[i]->SetState( fM->fBoolParameters[i].fValue ? kButtonDown : kButtonUp);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveParamListEditor::DoIntUpdate()
{
   // Slot for integer parameter update.

   TGNumberEntry *widget = (TGNumberEntry*) gTQSender;
   Int_t id = widget->WidgetId();
   if (id < 0 || id >= (int) fM->fIntParameters.size()) return;
   fM->fIntParameters[id].fValue = widget->GetNumberEntry()->GetIntNumber();

   fM->ParamChanged(fM->fIntParameters[id].fName);
   gTQSender = (void*) widget;
}

//______________________________________________________________________________
void TEveParamListEditor::DoFloatUpdate()
{
   // Slot for float parameter update.

   TGNumberEntry *widget = (TGNumberEntry*) gTQSender;
   Int_t id = widget->WidgetId();
   if (id < 0 || id >= (int) fM->fFloatParameters.size()) return;
   fM->fFloatParameters[id].fValue = widget->GetNumberEntry()->GetNumber();

   fM->ParamChanged(fM->fFloatParameters[id].fName);
   gTQSender = (void*) widget;
}

//______________________________________________________________________________
void TEveParamListEditor::DoBoolUpdate()
{
   // Slot for bool parameter update.

   TGCheckButton *widget = (TGCheckButton*) gTQSender;
   Int_t id = widget->WidgetId();
   if (id < 0 || id >= (int) fM->fBoolParameters.size()) return;
   fM->fBoolParameters[id].fValue = widget->IsOn();

   fM->ParamChanged(fM->fBoolParameters[id].fName);
   gTQSender = (void*) widget;
}
