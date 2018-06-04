// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TEveViewer.hxx"
#include "ROOT/TEveScene.hxx"
#include "ROOT/TEveSceneInfo.hxx"

#include "ROOT/TEveManager.hxx"
#include "ROOT/TEveSelection.hxx"

#include "TApplication.h"
#include "TEnv.h"
#include "TSystem.h"

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

/** \class TEveViewer
\ingroup TEve
Eve representation of a GL view. In a gist, it's a camera + a list of scenes.

*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.
///
/// The base-class TEveWindowFrame is constructed without a frame so
/// a default composite-frame is instantiated and stored in fGUIFrame.
/// Cleanup is set to no-cleanup as viewers need to be zapped with some
/// more care.

TEveViewer::TEveViewer(const char* n, const char* t) :
   TEveElementList(n, t)
{
   // SetChildClass(TEveSceneInfo::Class());
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TEveViewer::~TEveViewer()
{}

////////////////////////////////////////////////////////////////////////////////
/// Redraw viewer immediately.

void TEveViewer::Redraw(Bool_t /*resetCameras*/)
{
   // if (resetCameras) fGLViewer->PostSceneBuildSetup(kTRUE);
   // fGLViewer->RequestDraw(TGLRnrCtx::kLODHigh);
}

////////////////////////////////////////////////////////////////////////////////
/// Add 'scene' to the list of scenes.

void TEveViewer::AddScene(TEveScene* scene)
{
   static const TEveException eh("TEveViewer::AddScene ");

   for (auto i = BeginChildren(); i != EndChildren(); ++i)
   {
      auto sinfo = dynamic_cast<TEveSceneInfo*>(*i);

      if (sinfo && sinfo->GetScene() == scene)
      {
         throw eh + "scene already in the viewer.";
      }
   }

   TEveSceneInfo* si = new TEveSceneInfo(this, scene);
   AddElement(si);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove element 'el' from the list of children and also remove
/// appropriate GLScene from GLViewer's list of scenes.
/// Virtual from TEveElement.

void TEveViewer::RemoveElementLocal(TEveElement* /*el*/)
{
   // fGLViewer->RemoveScene(((TEveSceneInfo*)el)->GetGLScene());

   // XXXXX Notify clients !!! Or will this be automatic?
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all children, forwarded to GLViewer.
/// Virtual from TEveElement.

void TEveViewer::RemoveElementsLocal()
{
   // fGLViewer->RemoveAllScenes();

   // XXXXX Notify clients !!! Or will this be automatic?
}

////////////////////////////////////////////////////////////////////////////////
/// Object to be edited when this is selected, returns the TGLViewer.
/// Virtual from TEveElement.

TObject* TEveViewer::GetEditorObject(const TEveException& /*eh*/) const
{
   // if (!fGLViewer)
   //    throw(eh + "fGLViewer not set.");
   // return fGLViewer;

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Receive a pasted object. TEveViewer only accepts objects of
/// class TEveScene.
/// Virtual from TEveElement.

Bool_t TEveViewer::HandleElementPaste(TEveElement* el)
{
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


/** \class TEveViewerList
\ingroup TEve
List of Viewers providing common operations on TEveViewer collections.
*/

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TEveViewerList::~TEveViewerList()
{
   Disconnect();
}

////////////////////////////////////////////////////////////////////////////////
/// Call base-class implementation.
/// If compound is open and compound of the new element is not set,
/// the el's compound is set to this.

void TEveViewerList::AddElement(TEveElement* el)
{
   TEveElementList::AddElement(el);
   el->IncParentIgnoreCnt();
}

////////////////////////////////////////////////////////////////////////////////
/// Decompoundofy el, call base-class version.

void TEveViewerList::RemoveElementLocal(TEveElement* el)
{
   // This was needed as viewer was in EveWindowManager hierarchy, too.
   // el->DecParentIgnoreCnt();

   TEveElementList::RemoveElementLocal(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Decompoundofy children, call base-class version.

void TEveViewerList::RemoveElementsLocal()
{
   // This was needed as viewer was in EveWindowManager hierarchy, too.
   // el->DecParentIgnoreCnt();
   // for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   // {
   //    (*i)->DecParentIgnoreCnt();
   // }

   TEveElementList::RemoveElementsLocal();
}

////////////////////////////////////////////////////////////////////////////////
/// Connect to TGLViewer class-signals.

void TEveViewerList::Connect()
{
   // TQObject::Connect("TGLViewer", "MouseOver(TObject*,UInt_t)",
   //                   "TEveViewerList", this, "OnMouseOver(TObject*,UInt_t)");

   // TQObject::Connect("TGLViewer", "ReMouseOver(TObject*,UInt_t)",
   //                   "TEveViewerList", this, "OnReMouseOver(TObject*,UInt_t)");

   // TQObject::Connect("TGLViewer", "UnMouseOver(TObject*,UInt_t)",
   //                   "TEveViewerList", this, "OnUnMouseOver(TObject*,UInt_t)");

   // TQObject::Connect("TGLViewer", "Clicked(TObject*,UInt_t,UInt_t)",
   //                   "TEveViewerList", this, "OnClicked(TObject*,UInt_t,UInt_t)");

   // TQObject::Connect("TGLViewer", "ReClicked(TObject*,UInt_t,UInt_t)",
   //                   "TEveViewerList", this, "OnReClicked(TObject*,UInt_t,UInt_t)");

   // TQObject::Connect("TGLViewer", "UnClicked(TObject*,UInt_t,UInt_t)",
   //                   "TEveViewerList", this, "OnUnClicked(TObject*,UInt_t,UInt_t)");
}

////////////////////////////////////////////////////////////////////////////////
/// Disconnect from TGLViewer class-signals.

void TEveViewerList::Disconnect()
{
   // TQObject::Disconnect("TGLViewer", "MouseOver(TObject*,UInt_t)",
   //                      this, "OnMouseOver(TObject*,UInt_t)");

   // TQObject::Disconnect("TGLViewer", "ReMouseOver(TObject*,UInt_t)",
   //                      this, "OnReMouseOver(TObject*,UInt_t)");

   // TQObject::Disconnect("TGLViewer", "UnMouseOver(TObject*,UInt_t)",
   //                      this, "OnUnMouseOver(TObject*,UInt_t)");

   // TQObject::Disconnect("TGLViewer", "Clicked(TObject*,UInt_t,UInt_t)",
   //                      this, "OnClicked(TObject*,UInt_t,UInt_t)");

   // TQObject::Disconnect("TGLViewer", "ReClicked(TObject*,UInt_t,UInt_t)",
   //                      this, "OnReClicked(TObject*,UInt_t,UInt_t)");

   // TQObject::Disconnect("TGLViewer", "UnClicked(TObject*,UInt_t,UInt_t)",
   //                      this, "OnUnClicked(TObject*,UInt_t,UInt_t)");
}

////////////////////////////////////////////////////////////////////////////////
/// Repaint viewers that are tagged as changed.

void TEveViewerList::RepaintChangedViewers(Bool_t /*resetCameras*/, Bool_t /*dropLogicals*/)
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      // TGLViewer* glv = ((TEveViewer*)*i)->GetGLViewer();
      // if (glv->IsChanged())
      // {
      //    if (resetCameras) glv->PostSceneBuildSetup(kTRUE);
      //    if (dropLogicals) glv->SetSmartRefresh(kFALSE);

      //    glv->RequestDraw(TGLRnrCtx::kLODHigh);

      //    if (dropLogicals) glv->SetSmartRefresh(kTRUE);
      // }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Repaint all viewers.

void TEveViewerList::RepaintAllViewers(Bool_t /*resetCameras*/, Bool_t /*dropLogicals*/)
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      // TGLViewer* glv = ((TEveViewer*)*i)->GetGLViewer();

      // if (resetCameras) glv->PostSceneBuildSetup(kTRUE);
      // if (dropLogicals) glv->SetSmartRefresh(kFALSE);

      // glv->RequestDraw(TGLRnrCtx::kLODHigh);

      // if (dropLogicals) glv->SetSmartRefresh(kTRUE);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Delete annotations from all viewers.

void TEveViewerList::DeleteAnnotations()
{
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      // TGLViewer* glv = ((TEveViewer*)*i)->GetGLViewer();
      // glv->DeleteOverlayAnnotations();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Callback done from a TEveScene destructor allowing proper
/// removal of the scene from affected viewers.

void TEveViewerList::SceneDestructing(TEveScene* scene)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Show / hide tooltip for various MouseOver events.
/// Must be called from slots where sender is TGLEventHandler.

void TEveViewerList::HandleTooltip()
{
   if (fShowTooltip)
   {
      // TGLViewer       *glw = dynamic_cast<TGLViewer*>((TQObject*) gTQSender);
      // TGLEventHandler *glh = (TGLEventHandler*) glw->GetEventHandler();
      // if (REX::gEve->GetHighlight()->NumChildren() == 1)
      // {
      //    TString title(REX::gEve->GetHighlight()->FirstChild()->GetHighlightTooltip());
      //    if ( ! title.IsNull())
      //       glh->TriggerTooltip(title);
      // }
      // else
      // {
      //    glh->RemoveTooltip();
      // }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Slot for global TGLViewer::MouseOver() signal.
///
/// The attempt is made to determine the TEveElement being
/// represented by the physical shape and global highlight is updated
/// accordingly.
///
/// If TEveElement::IsPickable() returns false, the element is not
/// highlighted.
///
/// Highlight is always in single-selection mode.

void TEveViewerList::OnMouseOver(TObject *obj, UInt_t /*state*/)
{
   TEveElement *el = dynamic_cast<TEveElement*>(obj);
   if (el && !el->IsPickable())
      el = 0;

   // void *qsender = gTQSender;
   // REX::gEve->GetHighlight()->UserPickedElement(el, kFALSE);
   // gTQSender = qsender;

   HandleTooltip();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot for global TGLViewer::ReMouseOver().
///
/// The obj is dyn-casted to the TEveElement and global selection is
/// updated accordingly.
///
/// If TEveElement::IsPickable() returns false, the element is not
/// selected.

void TEveViewerList::OnReMouseOver(TObject *obj, UInt_t /*state*/)
{
   TEveElement* el = dynamic_cast<TEveElement*>(obj);
   if (el && !el->IsPickable())
      el = 0;

   // void *qsender = gTQSender;
   // REX::gEve->GetHighlight()->UserRePickedElement(el);
   // gTQSender = qsender;

   HandleTooltip();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot for global TGLViewer::UnMouseOver().
///
/// The obj is dyn-casted to the TEveElement and global selection is
/// updated accordingly.
///
/// If TEveElement::IsPickable() returns false, the element is not
/// selected.

void TEveViewerList::OnUnMouseOver(TObject *obj, UInt_t /*state*/)
{
   TEveElement* el = dynamic_cast<TEveElement*>(obj);
   if (el && !el->IsPickable())
      el = 0;

   // void *qsender = gTQSender;
   // REX::gEve->GetHighlight()->UserUnPickedElement(el);
   // gTQSender = qsender;

   HandleTooltip();
}

////////////////////////////////////////////////////////////////////////////////
/// Slot for global TGLViewer::Clicked().
///
/// The obj is dyn-casted to the TEveElement and global selection is
/// updated accordingly.
///
/// If TEveElement::IsPickable() returns false, the element is not
/// selected.

void TEveViewerList::OnClicked(TObject *obj, UInt_t /*button*/, UInt_t state)
{
   TEveElement* el = dynamic_cast<TEveElement*>(obj);
   if (el && !el->IsPickable())
      el = 0;
   REX::gEve->GetSelection()->UserPickedElement(el, state & kKeyControlMask);
}

////////////////////////////////////////////////////////////////////////////////
/// Slot for global TGLViewer::ReClicked().
///
/// The obj is dyn-casted to the TEveElement and global selection is
/// updated accordingly.
///
/// If TEveElement::IsPickable() returns false, the element is not
/// selected.

void TEveViewerList::OnReClicked(TObject *obj, UInt_t /*button*/, UInt_t /*state*/)
{
   TEveElement* el = dynamic_cast<TEveElement*>(obj);
   if (el && !el->IsPickable())
      el = 0;
   REX::gEve->GetSelection()->UserRePickedElement(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Slot for global TGLViewer::UnClicked().
///
/// The obj is dyn-casted to the TEveElement and global selection is
/// updated accordingly.
///
/// If TEveElement::IsPickable() returns false, the element is not
/// selected.

void TEveViewerList::OnUnClicked(TObject *obj, UInt_t /*button*/, UInt_t /*state*/)
{
   TEveElement* el = dynamic_cast<TEveElement*>(obj);
   if (el && !el->IsPickable())
      el = 0;
   REX::gEve->GetSelection()->UserUnPickedElement(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Set color brightness.

void TEveViewerList::SetColorBrightness(Float_t b)
{
   TEveUtil::SetColorBrightness(b, 1);
}

////////////////////////////////////////////////////////////////////////////////
/// Switch background color.

void TEveViewerList::SwitchColorSet()
{
   fUseLightColorSet = ! fUseLightColorSet;
   for (List_i i=fChildren.begin(); i!=fChildren.end(); ++i)
   {
      // TGLViewer* glv = ((TEveViewer*)*i)->GetGLViewer();
      // if ( fUseLightColorSet)
      //    glv->UseLightColorSet();
      // else
      //    glv->UseDarkColorSet();

      // glv->RequestDraw(TGLRnrCtx::kLODHigh);
   }
}
