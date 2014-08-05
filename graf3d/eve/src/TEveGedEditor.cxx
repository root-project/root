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

#include "TGButton.h"
#include "TGLabel.h"
#include "TGToolTip.h"
#include "TGDNDManager.h"
#include "TGMsgBox.h"

#include "TClass.h"
#include "TContextMenu.h"

//==============================================================================
// TEveGedEditor
//==============================================================================

//______________________________________________________________________________
//
// Specialization of TGedEditor for proper update propagation to
// TEveManager.

ClassImp(TEveGedEditor);

Int_t   TEveGedEditor::fgMaxExtraEditors = 10;
TList  *TEveGedEditor::fgExtraEditors    = new TList;

TContextMenu *TEveGedEditor::fgContextMenu = 0;

//______________________________________________________________________________
TEveGedEditor::TEveGedEditor(TCanvas* canvas, UInt_t width, UInt_t height) :
   TGedEditor(canvas, width, height),
   fElement  (0),
   fObject   (0)
{
   // Constructor.

   // Remove old name-frame -- it is created in TGedEditor constructor
   // so virtuals are not active yet.
   fTabContainer->RemoveAll();
   TGedFrame* nf = CreateNameFrame(fTabContainer, "Style");
   nf->SetGedEditor(this);
   nf->SetModelClass(0);
   fTabContainer->AddFrame(nf, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 2, 2, 2, 2));

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
TEveGedEditor::~TEveGedEditor()
{
   // Destructor.

   if (gDebug > 0)
      Info("TEveGedEditor::~TEveGedEditor", "%p going down.", this);
}

//______________________________________________________________________________
void TEveGedEditor::CloseWindow()
{
   // Called from window-manger close button.
   // Unregister from global list and delete the window.

   if (gDebug > 0)
      Info("TEveGedEditor::CloseWindow", "%p closing.", this);

   fgExtraEditors->Remove(this);

   DeleteWindow();
}

//______________________________________________________________________________
void TEveGedEditor::DeleteWindow()
{
   // This is exact clone of TGFrame::DeleteWindow().
   // Needs to be overriden together with CloseWindow() otherwise CINT
   // goes kaboom in timer execution.

   if (gDebug > 0)
      Info("TEveGedEditor::DeleteWindow", "%p shooting timer.", this);

   DisplayElement(0);

   if (gDNDManager) {
      if (gDNDManager->GetMainFrame() == this)
         gDNDManager->SetMainFrame(0);
   }
   if (!TestBit(kDeleteWindowCalled))
      TTimer::SingleShot(150, IsA()->GetName(), this, "ReallyDelete()");
   SetBit(kDeleteWindowCalled);
}

//______________________________________________________________________________
TGedFrame* TEveGedEditor::CreateNameFrame(const TGWindow* parent, const char* /*tab_name*/)
{
   // Create name-frame for a tab.

   return new TEveGedNameFrame(parent);
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

//==============================================================================

//______________________________________________________________________________
void TEveGedEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t event)
{
   // Set model object.

   fElement = dynamic_cast<TEveElement*>(obj);
   fObject  = obj;
   TGedEditor::SetModel(pad, obj, event);
}

//______________________________________________________________________________
void TEveGedEditor::Update(TGedFrame* /*gframe*/)
{
   // Virtual method from TGedEditor, called on every change.
   // Propagates changes to TEveElement and TEveManager.

   if (fElement)
   {
      fElement->ElementChanged();
      fElement->PropagateVizParamsToProjecteds();
   }

   gEve->Redraw3D();
}

//------------------------------------------------------------------------------
// Static functions for management of extra editors.
//------------------------------------------------------------------------------

//______________________________________________________________________________
void TEveGedEditor::SpawnNewEditor(TObject* obj)
{
   // Static function to create a new extra editor.

   if (fgExtraEditors->GetSize() >= fgMaxExtraEditors)
   {
      new TGMsgBox(gClient->GetDefaultRoot(), gEve->GetMainWindow(),
                   "Clutter warning",
                   "Maximum number of extra editors reached.",
                   kMBIconStop, kMBOk);
   }

   if (obj)
   {
      TEveGedEditor *ed = new TEveGedEditor();
      ed->DisplayObject(obj);
      ed->SetWindowName(Form("GED %s", obj->GetName()));

      fgExtraEditors->Add(ed);
   }
}

//______________________________________________________________________________
void TEveGedEditor::ElementChanged(TEveElement* el)
{
   // Element was changed. Update editors showing it.

   TObject *eobj = el->GetEditorObject("TEveGedEditor::ElementChanged ");
   TObjLink *lnk = fgExtraEditors->FirstLink();
   while (lnk)
   {
      TEveGedEditor *ed = (TEveGedEditor*) lnk->GetObject();
      if (ed->GetModel() == eobj)
         ed->DisplayElement(el);
      lnk = lnk->Next();
   }
}

//______________________________________________________________________________
void TEveGedEditor::ElementDeleted(TEveElement* el)
{
   // Element is being deleted. Close editors showing it.

   TObject *eobj = el->GetEditorObject("TEveGedEditor::ElementChanged ");
   TObjLink *lnk = fgExtraEditors->FirstLink();
   while (lnk)
   {
      TEveGedEditor *ed = (TEveGedEditor*) lnk->GetObject();
      if (ed->GetModel() == eobj)
      {
         TObjLink *next = lnk->Next();
         ed->DeleteWindow();
         fgExtraEditors->Remove(lnk);
         lnk = next;
      }
      else
      {
         lnk = lnk->Next();
      }
   }
}

//______________________________________________________________________________
void TEveGedEditor::DestroyEditors()
{
   // Destroys all editors. Called from EVE termination.

   while ( ! fgExtraEditors->IsEmpty())
   {
      TEveGedEditor *ed = (TEveGedEditor*) fgExtraEditors->First();
      ed->DeleteWindow();
      fgExtraEditors->RemoveFirst();
   }
}

//______________________________________________________________________________
TContextMenu* TEveGedEditor::GetContextMenu()
{
   // Return context menu object shared among eve-ged-editors.

   if (fgContextMenu == 0)
      fgContextMenu = new TContextMenu("", "");
   return fgContextMenu;
}


//==============================================================================
// TEveGedNameFrame
//==============================================================================

//______________________________________________________________________________
//
// Specialization of TGedNameFrame used in EVE.
// It provides the ability to undock given editor for easier use.
// Support for that is also provided from the TEveManager.

ClassImp(TEveGedNameFrame);

//______________________________________________________________________________
TEveGedNameFrame::TEveGedNameFrame(const TGWindow *p, Int_t width, Int_t height,
                                   UInt_t options) :
   TGedFrame(p, width, height, options),
   fNCButton(0)
{
   // Constructor.

   fNCButton = new TEveGedNameTextButton(this);
   fNCButton->SetTextColor(0x0020a0);
   AddFrame(fNCButton, new TGLayoutHints(kLHintsNormal | kLHintsExpandX));
   fNCButton->Connect("Clicked()", "TEveGedNameFrame", this, "SpawnEditorClone()");
}

//______________________________________________________________________________
TEveGedNameFrame::~TEveGedNameFrame()
{
   // Destructor.
}

//______________________________________________________________________________
void TEveGedNameFrame::SetModel(TObject* obj)
{
   // Set model object.

   if (obj)
   {
      fNCButton->SetText(Form("%s [%s]", obj->GetName(), obj->ClassName()));
      fNCButton->SetToolTipText(obj->GetTitle());
      fNCButton->SetEnabled(kTRUE);
   }
   else
   {
      fNCButton->SetText("No object selected.");
      fNCButton->SetToolTipText(0);
      fNCButton->SetEnabled(kFALSE);
   }
}

//______________________________________________________________________________
void TEveGedNameFrame::SpawnEditorClone()
{
   // Create a new floating editor with current object.

   TEveGedEditor::SpawnNewEditor(fGedEditor->GetModel());
}


//==============================================================================
// TEveGedNameTextButton
//==============================================================================

//______________________________________________________________________________
//
// Specialization of TGTextButton for EVE name frame.
// It opens a context-menu on right-click.

ClassImp(TEveGedNameTextButton);

//______________________________________________________________________________
TEveGedNameTextButton::TEveGedNameTextButton(TEveGedNameFrame* p) :
   TGTextButton(p, ""),
   fFrame(p)
{
   // Constructor.

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask,
                         kNone, kNone);
}

//______________________________________________________________________________
TEveGedNameTextButton::~TEveGedNameTextButton()
{
   // Destructor.
}

//______________________________________________________________________________
Bool_t TEveGedNameTextButton::HandleButton(Event_t* event)
{
   // Handle button.

   static const TEveException eh("TEveGedNameTextButton::HandleButton ");

   if (fTip) fTip->Hide();
   if (fState == kButtonDisabled) return kTRUE;

   if (event->fCode == kButton3 && event->fType == kButtonPress)
   {
      TEveGedEditor *eged = (TEveGedEditor*) fFrame->GetGedEditor();
      TEveElement   *el   = eged->GetEveElement();
      if (el)
         TEveGedEditor::GetContextMenu()->Popup(event->fXRoot, event->fYRoot,
                                                el->GetObject(eh));
      return 1;
   }
   else if (event->fCode == kButton1)
   {
      return TGTextButton::HandleButton(event);
   }
   else
   {
      return 0;
   }
}
