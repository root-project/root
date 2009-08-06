// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveGedEditor.h"
#include "TEveElement.h"
#include "TEveManager.h"

#include "TGedFrame.h"

#include "TClass.h"

//==============================================================================
// TEveGedEditor
//==============================================================================

//______________________________________________________________________________
//
// Specialization of TGedEditor for proper update propagation to
// TEveManager.

ClassImp(TEveGedEditor);

//______________________________________________________________________________
TEveGedEditor::TEveGedEditor(TCanvas* canvas, Int_t width, Int_t height) :
   TGedEditor(canvas),
   fElement  (0),
   fObject   (0)
{
   // Constructor.

   Resize(width, height);

   // Fix priority for TAttMarkerEditor.
   TClass* amClass = TClass::GetClass("TAttMarker");
   TClass* edClass = TClass::GetClass("TAttMarkerEditor");
   TGWindow *exroot = (TGWindow*) fClient->GetRoot();
   fClient->SetRoot(fTabContainer);
   SetFrameCreator(this);
   TGedFrame *frame = reinterpret_cast<TGedFrame*>(edClass->New());
   frame->SetModelClass(amClass);
   {
      Int_t off = edClass->GetDataMemberOffset("fPriority");
      if (off == 0)
         Warning("TEveGedEditor::TEveGedEditor", "Can't fix priority for TAttMarkerEditor.\n");
      else
         * (Int_t*) (((char*)frame) + off) = 1;
   }
   SetFrameCreator(0);
   fClient->SetRoot(exroot);
   fFrameMap.Add(amClass, frame);
}

//______________________________________________________________________________
TEveElement* TEveGedEditor::GetEveElement() const
{
   // Return eve-element if it is the model object.

   return (fModel == fObject) ? fElement : 0;
}

//______________________________________________________________________________
void TEveGedEditor::DisplayElement(TEveElement* re)
{
   // Show a TEveElement in editor.

   static const TEveException eh("TEveGedEditor::DisplayElement ");

   fElement = re;
   fObject  = fElement ? fElement->GetEditorObject(eh) : 0;
   TGedEditor::SetModel(fPad, fObject, kButton1Down);
}

//______________________________________________________________________________
void TEveGedEditor::DisplayObject(TObject* obj)
{
   // Show a TObject in editor.

   fElement = dynamic_cast<TEveElement*>(obj);
   fObject  = obj;
   TGedEditor::SetModel(fPad, obj, kButton1Down);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveGedEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t event)
{
   // Set model object.

   // !!!! do something so that such calls from elswhere will also
   // now the render element

   fElement = dynamic_cast<TEveElement*>(obj);
   fObject  = obj;
   TGedEditor::SetModel(pad, obj, event);
}

//______________________________________________________________________________
void TEveGedEditor::Update(TGedFrame* /*gframe*/)
{
   // Virtual method from TGedEditor, called on every change.
   // Propagates changes to TEveElement and TEveManager.

   if (fElement) {
      fElement->ElementChanged();
      fElement->PropagateVizParamsToProjecteds();
   }

   gEve->Redraw3D();
}

/******************************************************************************/

/*
// Attempt to enable mouse-wheel in geditor -- failed.
Bool_t TEveGedEditor::HandleButton(Event_t *event)
{
// Handle mouse button event in container.

printf("odfjgsf\n");
if (event->fCode == kButton4 || event->fCode == kButton5) {
return fCan->GetContainer()->HandleButton(event);
} else {
return TGedEditor::HandleButton(event);
}
}
*/
