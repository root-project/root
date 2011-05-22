// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveViewer.h"
#include "TEveScene.h"
#include "TEveSceneInfo.h"

#include "TEveManager.h"
#include "TEveSelection.h"

#include "TGLFormat.h"
#include "TGLSAViewer.h"
#include "TGLEmbeddedViewer.h"
#include "TGLScenePad.h"

#include "TGLEventHandler.h"

#include "TApplication.h"
#include "TEnv.h"
#include "TSystem.h"

//==============================================================================
//==============================================================================
// TEveViewer
//==============================================================================

//______________________________________________________________________________
//
// Eve representation of TGLViewer.
//
// The gl-viewer is owned by this class and is deleted in destructor.
//
// The frame is not deleted, it is expected that the gl-viewer implementation
// will delete that. TGLSAViewer and TGEmbeddedViewer both do so.
// This could be an optional argument to SetGLViewer. A frame could be
// passed as well.
//
// When stand-alone viewer is requested, it will come up with menu-hiding
// enabled by default. If you dislike this, add the following line to rootrc
// file (or set corresponding gEnv entry in application initialization):
//   Eve.Viewer.HideMenus: off

ClassImp(TEveViewer);

Bool_t TEveViewer::fgInitInternal        = kFALSE;
Bool_t TEveViewer::fgRecreateGlOnDockOps = kFALSE;

//______________________________________________________________________________
TEveViewer::TEveViewer(const char* n, const char* t) :
   TEveWindowFrame(0, n, t),
   fGLViewer      (0),
   fGLViewerFrame (0)
{
   // Constructor.
   // The base-class TEveWindowFrame is constructed without a frame so
   // a default composite-frame is instantiated and stored in fGUIFrame.
   // Cleanup is set to no-cleanup as viewers need to be zapped with some
   // more care.

   SetChildClass(TEveSceneInfo::Class());
   fGUIFrame->SetCleanup(kNoCleanup); // the gl-viewer's frame deleted elsewhere.

   if (!fgInitInternal)
   {
      InitInternal();
   }
}

//______________________________________________________________________________
TEveViewer::~TEveViewer()
{
   // Destructor.

   fGLViewer->SetEventHandler(0);

   fGLViewerFrame->UnmapWindow();
   GetGUICompositeFrame()->RemoveFrame(fGLViewerFrame);
   fGLViewerFrame->ReparentWindow(gClient->GetDefaultRoot());
   TTimer::SingleShot(150, "TGLViewer", fGLViewer, "Delete()");
}

/******************************************************************************/

//______________________________________________________________________________
void TEveViewer::InitInternal()
{
   // Initialize static data-members according to running conditions.

   // Determine if display is running on a mac.
   // This also works for ssh connection mac->linux.
   fgRecreateGlOnDockOps = (gVirtualX->SupportsExtension("Apple-WM") == 1);

   fgInitInternal = kTRUE;
}

//______________________________________________________________________________
void TEveViewer::PreUndock()
{
   // Virtual function called before a window is undocked.
   // On mac we have to force recreation of gl-context.

   TEveWindowFrame::PreUndock();
   if (fgRecreateGlOnDockOps)
   {
      // Mac only: TGLWidget can be already deleted
      // in case of recursive delete
      if (fGLViewer->GetGLWidget())
      {
         fGLViewer->DestroyGLWidget();
      }
   }
}

//______________________________________________________________________________
void TEveViewer::PostDock()
{
   // Virtual function called after a window is docked.
   // On mac we have to force recreation of gl-context.

   if (fgRecreateGlOnDockOps) {
      fGLViewer->CreateGLWidget();
   }
   TEveWindowFrame::PostDock();
}

/******************************************************************************/

//______________________________________________________________________________
const TGPicture* TEveViewer::GetListTreeIcon(Bool_t)
{
   // Return TEveViewer icon.

   return TEveElement::fgListTreeIcons[1];
}

//______________________________________________________________________________
void TEveViewer::SetGLViewer(TGLViewer* viewer, TGFrame* frame)
{
   // Set TGLViewer that is represented by this object.
   // The old gl-viewer is deleted.

   delete fGLViewer;
   fGLViewer      = viewer;
   fGLViewerFrame = frame;

   fGLViewer->SetSmartRefresh(kTRUE);
}

//______________________________________________________________________________
TGLSAViewer* TEveViewer::SpawnGLViewer(TGedEditor* ged, Bool_t stereo)
{
   // Spawn new GLViewer and adopt it.

   static const TEveException kEH("TEveViewer::SpawnGLViewer ");

   TGCompositeFrame* cf = GetGUICompositeFrame();

   TGLFormat *form = 0;
   if (stereo)
   {
      form = new TGLFormat;
      form->SetStereo(kTRUE);
   }

   cf->SetEditable(kTRUE);
   TGLSAViewer* v = 0;
   try
   {
      v = new TGLSAViewer(cf, 0, ged, form);
   }
   catch (std::exception&)
   {
      Error("SpawnGLViewer", "Insufficient support from the graphics hardware. Aborting.");
      gApplication->Terminate(1);
   }
   cf->SetEditable(kFALSE);
   v->ToggleEditObject();
   v->DisableCloseMenuEntries();
   if (gEnv->GetValue("Eve.Viewer.HideMenus", 1) == 1)
   {
      v->EnableMenuBarHiding();
   }
   SetGLViewer(v, v->GetFrame());

   if (stereo)
      v->SetStereo(kTRUE);

   if (fEveFrame == 0)
      PreUndock();

   return v;
}

//______________________________________________________________________________
TGLEmbeddedViewer* TEveViewer::SpawnGLEmbeddedViewer(TGedEditor* ged, Int_t border)
{
   // Spawn new GLViewer and adopt it.

   static const TEveException kEH("TEveViewer::SpawnGLEmbeddedViewer ");

   TGCompositeFrame* cf = GetGUICompositeFrame();

   TGLEmbeddedViewer* v = new TGLEmbeddedViewer(cf, 0, ged, border);
   SetGLViewer(v, v->GetFrame());

   cf->AddFrame(fGLViewerFrame, new TGLayoutHints(kLHintsNormal | kLHintsExpandX | kLHintsExpandY));

   fGLViewerFrame->MapWindow();

   if (fEveFrame == 0)
      PreUndock();

   return v;
}

//______________________________________________________________________________
void TEveViewer::Redraw(Bool_t resetCameras)
{
   // Redraw viewer immediately.

   if (resetCameras) fGLViewer->PostSceneBuildSetup(kTRUE);
   fGLViewer->RequestDraw(TGLRnrCtx::kLODHigh);
}

//______________________________________________________________________________
void TEveViewer::SwitchStereo()
{
   // Switch stereo mode.
   // This only works TGLSAViewers and, of course, with stereo support
   // provided by the OpenGL driver.

   TGLSAViewer *v = dynamic_cast<TGLSAViewer*>(fGLViewer);

   if (!v) {
      Warning("SwitchStereo", "Only supported for TGLSAViewer.");
      return;
   }

   v->DestroyGLWidget();
   TGLFormat *f = v->GetFormat();
switch_stereo:
   f->SetStereo(!f->IsStereo());
   v->SetStereo(f->IsStereo());
   try
   {
      v->CreateGLWidget();
   }
   catch (std::exception&)
   {
      Error("SwitchStereo", "Insufficient support from the graphics hardware. Reverting.");
      goto switch_stereo;
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveViewer::AddScene(TEveScene* scene)
{
   // Add 'scene' to the list of scenes.

   static const TEveException eh("TEveViewer::AddScene ");

   TGLSceneInfo* glsi = fGLViewer->AddScene(scene->GetGLScene());
   if (glsi != 0) {
      TEveSceneInfo* si = new TEveSceneInfo(this, scene, glsi);
      AddElement(si);
   } else {
      throw(eh + "scene already in the viewer.");
   }
}

//______________________________________________________________________________
void TEveViewer::RemoveElementLocal(TEveElement* el)
{
   // Remove element 'el' from the list of children and also remove
   // appropriate GLScene from GLViewer's list of scenes.
   // Virtual from TEveElement.

   fGLViewer->RemoveScene(((TEveSceneInfo*)el)->GetGLScene());
}

//______________________________________________________________________________
void TEveViewer::RemoveElementsLocal()
{
   // Remove all children, forwarded to GLViewer.
   // Virtual from TEveElement.

   fGLViewer->RemoveAllScenes();
}

//______________________________________________________________________________
TObject* TEveViewer::GetEditorObject(const TEveException& eh) const
{
   // Object to be edited when this is selected, returns the TGLViewer.
   // Virtual from TEveElement.

   if (!fGLViewer)
      throw(eh + "fGLViewer not set.");
   return fGLViewer;
}

//______________________________________________________________________________
Bool_t TEveViewer::HandleElementPaste(TEveElement* el)
{
   // Receive a pasted object. TEveViewer only accepts objects of
   // class TEveScene.
   // Virtual from TEveElement.

   static const TEveException eh("TEveViewer::HandleElementPaste ");

   TEveScene* scene = dynamic_cast<TEveScene*>(el);
   if (scene != 0) {
      AddScene(scene);
      return kTRUE;
   } else {
      Warning(eh.Data(), "class TEveViewer only accepts TEveScene paste argument.");
      return kFALSE;
   }
}


/******************************************************************************/
/******************************************************************************/
// TEveViewerList
/******************************************************************************/

//______________________________________________________________________________
//
// List of Viewers providing common operations on TEveViewer collections.

ClassImp(TEveViewerList);

//______________________________________________________________________________
TEveViewerList::TEveViewerList(const char* n, const char* t) :
   TEveElementList(n, t),
   fShowTooltip   (kTRUE),

   fBrightness(0),
   fUseLightColorSet(kFALSE)
{
   // Constructor.

   SetChildClass(TEveViewer::Class());
   Connect();
}

//______________________________________________________________________________
TEveViewerList::~TEveViewerList()
{
   // Destructor.

   Disconnect();
}

//==============================================================================

//______________________________________________________________________________
void TEveViewerList::AddElement(TEveElement* el)
{
   // Call base-class implementation.
   // If compund is open and compound of the new element is not set,
   // the el's compound is set to this.

   TEveElementList::AddElement(el);
   el->IncParentIgnoreCnt();
}

//______________________________________________________________________________
void TEveViewerList::RemoveElementLocal(TEveElement* el)
{
   // Decompoundofy el, call base-class version.

   el->DecParentIgnoreCnt();
   TEveElementList::RemoveElementLocal(el);
}

//______________________________________________________________________________
void TEveViewerList::RemoveElementsLocal()
{
   // Decompoundofy children, call base-class version.

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      (*i)->DecParentIgnoreCnt();
   }

   TEveElementList::RemoveElementsLocal();
}

//==============================================================================

//______________________________________________________________________________
void TEveViewerList::Connect()
{
   // Connect to TGLViewer class-signals.

   TQObject::Connect("TGLViewer", "MouseOver(TObject*,UInt_t)",
                     "TEveViewerList", this, "OnMouseOver(TObject*,UInt_t)");

   TQObject::Connect("TGLViewer", "ReMouseOver(TObject*,UInt_t)",
                     "TEveViewerList", this, "OnReMouseOver(TObject*,UInt_t)");

   TQObject::Connect("TGLViewer", "UnMouseOver(TObject*,UInt_t)",
                     "TEveViewerList", this, "OnUnMouseOver(TObject*,UInt_t)");

   TQObject::Connect("TGLViewer", "Clicked(TObject*,UInt_t,UInt_t)",
                     "TEveViewerList", this, "OnClicked(TObject*,UInt_t,UInt_t)");

   TQObject::Connect("TGLViewer", "ReClicked(TObject*,UInt_t,UInt_t)",
                     "TEveViewerList", this, "OnReClicked(TObject*,UInt_t,UInt_t)");

   TQObject::Connect("TGLViewer", "UnClicked(TObject*,UInt_t,UInt_t)",
                     "TEveViewerList", this, "OnUnClicked(TObject*,UInt_t,UInt_t)");
}

//______________________________________________________________________________
void TEveViewerList::Disconnect()
{
   // Disconnect from TGLViewer class-signals.

   TQObject::Disconnect("TGLViewer", "MouseOver(TObject*,UInt_t)",
                        this, "OnMouseOver(TObject*,UInt_t)");

   TQObject::Disconnect("TGLViewer", "ReMouseOver(TObject*,UInt_t)",
                        this, "OnReMouseOver(TObject*,UInt_t)");

   TQObject::Disconnect("TGLViewer", "UnMouseOver(TObject*,UInt_t)",
                        this, "OnUnMouseOver(TObject*,UInt_t)");

   TQObject::Disconnect("TGLViewer", "Clicked(TObject*,UInt_t,UInt_t)",
                        this, "OnClicked(TObject*,UInt_t,UInt_t)");

   TQObject::Disconnect("TGLViewer", "ReClicked(TObject*,UInt_t,UInt_t)",
                        this, "OnReClicked(TObject*,UInt_t,UInt_t)");

   TQObject::Disconnect("TGLViewer", "UnClicked(TObject*,UInt_t,UInt_t)",
                        this, "OnUnClicked(TObject*,UInt_t,UInt_t)");
}

/******************************************************************************/

//______________________________________________________________________________
void TEveViewerList::RepaintChangedViewers(Bool_t resetCameras, Bool_t dropLogicals)
{
   // Repaint viewers that are tagged as changed.

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TGLViewer* glv = ((TEveViewer*)*i)->GetGLViewer();
      if (glv->IsChanged())
      {
         // printf(" TEveViewer '%s' changed ... reqesting draw.\n", (*i)->GetObject()->GetName());

         if (resetCameras) glv->PostSceneBuildSetup(kTRUE);
         if (dropLogicals) glv->SetSmartRefresh(kFALSE);

         glv->RequestDraw(TGLRnrCtx::kLODHigh);

         if (dropLogicals) glv->SetSmartRefresh(kTRUE);
      }
   }
}

//______________________________________________________________________________
void TEveViewerList::RepaintAllViewers(Bool_t resetCameras, Bool_t dropLogicals)
{
   // Repaint all viewers.

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TGLViewer* glv = ((TEveViewer*)*i)->GetGLViewer();

      // printf(" TEveViewer '%s' sending redraw reqest.\n", (*i)->GetObject()->GetName());

      if (resetCameras) glv->PostSceneBuildSetup(kTRUE);
      if (dropLogicals) glv->SetSmartRefresh(kFALSE);

      glv->RequestDraw(TGLRnrCtx::kLODHigh);

      if (dropLogicals) glv->SetSmartRefresh(kTRUE);
   }
}

//______________________________________________________________________________
void TEveViewerList::DeleteAnnotations()
{
   // Delete annotations from all viewers.

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TGLViewer* glv = ((TEveViewer*)*i)->GetGLViewer();
      glv->DeleteOverlayAnnotations();
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveViewerList::SceneDestructing(TEveScene* scene)
{
   // Callback done from a TEveScene destructor allowing proper
   // removal of the scene from affected viewers.

   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      TEveViewer* viewer = (TEveViewer*) *i;
      List_i j = viewer->BeginChildren();
      while (j != viewer->EndChildren())
      {
         TEveSceneInfo* sinfo = (TEveSceneInfo*) *j;
         ++j;
         if (sinfo->GetScene() == scene)
            viewer->RemoveElement(sinfo);
      }
   }
}


/******************************************************************************/
// Processing of events from TGLViewers.
/******************************************************************************/

//______________________________________________________________________________
void TEveViewerList::HandleTooltip()
{
   // Show / hide tooltip for various MouseOver events.
   // Must be called from slots where sender is TGLEventHandler.

   if (fShowTooltip)
   {
      TGLViewer       *glw = dynamic_cast<TGLViewer*>((TQObject*) gTQSender);
      TGLEventHandler *glh = (TGLEventHandler*) glw->GetEventHandler();
      if (gEve->GetHighlight()->NumChildren() == 1)
      {
         TString title(gEve->GetHighlight()->FirstChild()->GetHighlightTooltip());
         if ( ! title.IsNull())
            glh->TriggerTooltip(title);
      }
      else
      {
         glh->RemoveTooltip();
      }
   }
}

//______________________________________________________________________________
void TEveViewerList::OnMouseOver(TObject *obj, UInt_t /*state*/)
{
   // Slot for global TGLViewer::MouseOver() signal.
   //
   // The attempt is made to determine the TEveElement being
   // represented by the physical shape and global higlight is updated
   // accordingly.
   //
   // If TEveElement::IsPickable() returns false, the element is not
   // highlighted.
   //
   // Highlight is always in single-selection mode.

   TEveElement *el = dynamic_cast<TEveElement*>(obj);
   if (el && !el->IsPickable())
      el = 0;

   void *qsender = gTQSender;
   gEve->GetHighlight()->UserPickedElement(el, kFALSE);
   gTQSender = qsender;

   HandleTooltip();
}

//______________________________________________________________________________
void TEveViewerList::OnReMouseOver(TObject *obj, UInt_t /*state*/)
{
   // Slot for global TGLViewer::ReMouseOver().
   //
   // The obj is dyn-casted to the TEveElement and global selection is
   // updated accordingly.
   //
   // If TEveElement::IsPickable() returns false, the element is not
   // selected.

   TEveElement* el = dynamic_cast<TEveElement*>(obj);
   if (el && !el->IsPickable())
      el = 0;

   void *qsender = gTQSender;
   gEve->GetHighlight()->UserRePickedElement(el);
   gTQSender = qsender;

   HandleTooltip();
}

//______________________________________________________________________________
void TEveViewerList::OnUnMouseOver(TObject *obj, UInt_t /*state*/)
{
   // Slot for global TGLViewer::UnMouseOver().
   //
   // The obj is dyn-casted to the TEveElement and global selection is
   // updated accordingly.
   //
   // If TEveElement::IsPickable() returns false, the element is not
   // selected.

   TEveElement* el = dynamic_cast<TEveElement*>(obj);
   if (el && !el->IsPickable())
      el = 0;

   void *qsender = gTQSender;
   gEve->GetHighlight()->UserUnPickedElement(el);
   gTQSender = qsender;

   HandleTooltip();
}

//______________________________________________________________________________
void TEveViewerList::OnClicked(TObject *obj, UInt_t /*button*/, UInt_t state)
{
   // Slot for global TGLViewer::Clicked().
   //
   // The obj is dyn-casted to the TEveElement and global selection is
   // updated accordingly.
   //
   // If TEveElement::IsPickable() returns false, the element is not
   // selected.

   TEveElement* el = dynamic_cast<TEveElement*>(obj);
   if (el && !el->IsPickable())
      el = 0;
   gEve->GetSelection()->UserPickedElement(el, state & kKeyControlMask);
}

//______________________________________________________________________________
void TEveViewerList::OnReClicked(TObject *obj, UInt_t /*button*/, UInt_t /*state*/)
{
   // Slot for global TGLViewer::ReClicked().
   //
   // The obj is dyn-casted to the TEveElement and global selection is
   // updated accordingly.
   //
   // If TEveElement::IsPickable() returns false, the element is not
   // selected.

   TEveElement* el = dynamic_cast<TEveElement*>(obj);
   if (el && !el->IsPickable())
      el = 0;
   gEve->GetSelection()->UserRePickedElement(el);
}

//______________________________________________________________________________
void TEveViewerList::OnUnClicked(TObject *obj, UInt_t /*button*/, UInt_t /*state*/)
{
   // Slot for global TGLViewer::UnClicked().
   //
   // The obj is dyn-casted to the TEveElement and global selection is
   // updated accordingly.
   //
   // If TEveElement::IsPickable() returns false, the element is not
   // selected.

   TEveElement* el = dynamic_cast<TEveElement*>(obj);
   if (el && !el->IsPickable())
      el = 0;
   gEve->GetSelection()->UserUnPickedElement(el);
}

//______________________________________________________________________________
void TEveViewerList::SetColorBrightness(Float_t b)
{
   // Set color brightness.

   TEveUtil::SetColorBrightness(b, 1);
}

//______________________________________________________________________________
void TEveViewerList::SwitchColorSet()
{
   // Switch background color.

   fUseLightColorSet = ! fUseLightColorSet;
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {  
      TGLViewer* glv = ((TEveViewer*)*i)->GetGLViewer();
      if ( fUseLightColorSet)
         glv->UseLightColorSet();
      else 
         glv->UseDarkColorSet();

      glv->RequestDraw(TGLRnrCtx::kLODHigh);
   }
}
