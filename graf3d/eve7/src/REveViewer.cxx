// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveViewer.hxx>

#include <ROOT/REveUtil.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveSceneInfo.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveSelection.hxx>
#include <ROOT/REveText.hxx>

#include <nlohmann/json.hpp>
#include "TROOT.h"

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

/** \class REveViewer
\ingroup REve
Eve representation of a GL view. In a gist, it's a camera + a list of scenes.

*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveViewer::REveViewer(const std::string& n, const std::string& t) :
   REveElement(n, t)
{
   // SetChildClass(TClass::GetClass<REveSceneInfo>());
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

REveViewer::~REveViewer()
{}

////////////////////////////////////////////////////////////////////////////////
/// Redraw viewer immediately.

void REveViewer::Redraw(Bool_t /*resetCameras*/)
{
   // if (resetCameras) fGLViewer->PostSceneBuildSetup(kTRUE);
   // fGLViewer->RequestDraw(TGLRnrCtx::kLODHigh);
}

////////////////////////////////////////////////////////////////////////////////
/// Add 'scene' to the list of scenes.

void REveViewer::AddScene(REveScene *scene)
{
   static const REveException eh("REveViewer::AddScene ");

   for (auto &c: RefChildren()) {
      auto sinfo = dynamic_cast<REveSceneInfo*>(c);

      if (sinfo && sinfo->GetScene() == scene)
      {
         throw eh + "scene already in the viewer.";
      }
   }

   auto si = new REveSceneInfo(this, scene);
   AddElement(si);
}

////////////////////////////////////////////////////////////////////////////////
/// Remove element 'el' from the list of children and also remove
/// appropriate GLScene from GLViewer's list of scenes.
/// Virtual from REveElement.

void REveViewer::RemoveElementLocal(REveElement* /*el*/)
{
   // fGLViewer->RemoveScene(((REveSceneInfo*)el)->GetGLScene());

   // XXXXX Notify clients !!! Or will this be automatic?
}

////////////////////////////////////////////////////////////////////////////////
/// Remove all children, forwarded to GLViewer.
/// Virtual from REveElement.

void REveViewer::RemoveElementsLocal()
{
   // fGLViewer->RemoveAllScenes();

   // XXXXX Notify clients !!! Or will this be automatic?
}


/** \class REveViewerList
\ingroup REve
List of Viewers providing common operations on REveViewer collections.
*/

////////////////////////////////////////////////////////////////////////////////
//
void REveViewer::SetAxesType(int at)
{
   fAxesType = (EAxesType)at;
   if (fAxesType != kAxesNone) {
      std::string fn = "LiberationSerif-Regular";
      std::string rf_dir = std::string(TROOT::GetDataDir().Data()) + "/fonts/";
      REX::REveText::AssertSdfFont(fn, rf_dir + fn + ".ttf");
   }
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
//
void REveViewer::SetBlackBackground(bool x)
{
   fBlackBackground = x;
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Stream Camera Info.
/// Virtual from REveElement.
int REveViewer::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   fCamera.WriteCoreJson(j, rnr_offset);

   j["Mandatory"] = fMandatory;
   j["AxesType"] = fAxesType;
   j["BlackBg"] = fBlackBackground;

   j["UT_PostStream"] = "UT_EveViewerUpdate";

   return REveElement::WriteCoreJson(j, rnr_offset);
}

////////////////////////////////////////////////////////////////////////////////
/// Function called from MIR when user closes one of the viewer window.
//  Client id stored in thread local data
void REveViewer::DisconnectClient()
{
   gEve->DisconnectEveViewer(this);
}
////////////////////////////////////////////////////////////////////////////////
/// Function called from MIR when user wants to stream unsubscribed view.
//  Client id stored in thread local data
void REveViewer::ConnectClient()
{
   gEve->ConnectEveViewer(this);
}

////////////////////////////////////////////////////////////////////////////////
///
//  Set Flag if this viewer is presented on connect
void REveViewer::SetMandatory(bool x)
{
   fMandatory = x;
   for (auto &c : RefChildren()) {
      REveSceneInfo *sinfo = dynamic_cast<REveSceneInfo *>(c);
      sinfo->GetScene()->GetScene()->SetMandatory(fMandatory);
   }
}

////////////////////////////////////////////////////////////////////////////////
///
//  Set base vectors of camera base matrix
//
void REveViewer::SetCameraType(ECameraType cameraType)
{
   switch(cameraType) {
      case kCameraPerspXOZ:
         fCamera.Setup(kCameraPerspXOZ, "PerspXOZ", REveVector(-1.0, 0.0, 0.0), REveVector(0.0, 1.0, 0.0)); // XOZ floor
         break;
      case kCameraPerspYOZ:
         fCamera.Setup(kCameraPerspYOZ, "PerspYOZ", REveVector(0.0, -1.0, 0.0), REveVector(1.0, 0.0, 0.0));// YOZ floor
         break;
      case kCameraPerspXOY:
         fCamera.Setup(kCameraPerspXOY, "PerspXOY", REveVector(-1.0, 0.0, 0.0), REveVector(0.0, 0.0, 1.0));// XOY floor
         break;
      case kCameraOrthoXOY:
         fCamera.Setup(kCameraOrthoXOY, "OrthoXOY", REveVector(0.0, 0.0, 1.0), REveVector(0.0, 1.0, 0.0));// Looking down  Z axis,  X horz, Y vert
         break;
      case kCameraOrthoXOZ:
         fCamera.Setup(kCameraOrthoXOZ, "OrthoXOZ", REveVector(0.0, -1.0, 0.0), REveVector(0.0, 0.0, 1.0));// Looking along Y axis,  X horz, Z vert
         break;
      case kCameraOrthoZOY:
         fCamera.Setup(kCameraOrthoZOY, "OrthoZOY", REveVector(-1.0, 0.0, 0.0), REveVector(0.0, 1.0, 0.0));// Looking along X axis,  Z horz, Y vert
         break;
      case kCameraOrthoZOX:
         fCamera.Setup(kCameraOrthoZOX, "OrthoZOX", REveVector(0.0,-1.0, 0.0), REveVector(1.0, 0.0, 0.0)); // Looking along Y axis,  Z horz, X vert
         break;
      case kCameraOrthoXnOY:
         fCamera.Setup(kCameraOrthoXnOY,  "OrthoXnOY", REveVector(0.0, 0.0, -1.0), REveVector(0.0, 1.0, 0.0));// Looking along Z axis, -X horz, Y vert
         break;
      case kCameraOrthoXnOZ:
         fCamera.Setup(kCameraOrthoXnOZ, "OrthoXnOZ", REveVector(0.0, 1.0, 0.0), REveVector(0.0, 0.0, 1.0));// Looking down  Y axis, -X horz, Z vert
         break;
      case kCameraOrthoZnOY:
         fCamera.Setup(kCameraOrthoZnOY, "OrthoZnOY", REveVector(1.0, 0.0, 0.0), REveVector(0.0, 1.0, 0.0)); // Looking down  X axis, -Z horz, Y vert
         break;
      case kCameraOrthoZnOX:
         fCamera.Setup(kCameraOrthoZnOX, "OrthoZnOX", REveVector(0.0, 1.0, 0.0), REveVector(1.0, 0.0, 0.0)); // Looking down  Y axis, -Z horz, X vert
         break;
      default:
         Error("REveViewer::SetCurrentCamera", "invalid camera type");
         return;
   }
}

////////////////////////////////////////////////////////////////////////////////
//
//  Set camera base matrix
//
void REveViewer::REveCamera::Setup( ECameraType type, const std::string& name, REveVector v1, REveVector v2)
{
   fType = type;
   fName = name;
   fV1 = v1;
   fV2 = v2;
}

////////////////////////////////////////////////////////////////////////////////
///
//  Stream camera info
//
int REveViewer::REveCamera::WriteCoreJson(nlohmann::json &j, Int_t /*rnr_offset*/)
{
   nlohmann::json out;
   out["type"] = fName;
   out["V1"] = {fV1.fX, fV1.fY, fV1.fZ};
   out["V2"] = {fV2.fX, fV2.fY, fV2.fZ};

   j["camera"] = out;

   return 0;
}

////////////////////////////////////////////////////////////////////////////////

REveViewerList::REveViewerList(const std::string &n, const std::string &t) :
   REveElement  (n, t),
   fShowTooltip (kTRUE),

   fBrightness(0),
   fUseLightColorSet(kFALSE)
{
   // Constructor.

   SetChildClass(TClass::GetClass<REveViewer>());
   Connect();
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

REveViewerList::~REveViewerList()
{
   Disconnect();
}

////////////////////////////////////////////////////////////////////////////////
/// Call base-class implementation.
/// If compound is open and compound of the new element is not set,
/// the el's compound is set to this.

void REveViewerList::AddElement(REveElement* el)
{
   REveElement::AddElement(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Decompoundofy el, call base-class version.

void REveViewerList::RemoveElementLocal(REveElement* el)
{
   // This was needed as viewer was in EveWindowManager hierarchy, too.
   // el->DecParentIgnoreCnt();

   REveElement::RemoveElementLocal(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Decompoundofy children, call base-class version.

void REveViewerList::RemoveElementsLocal()
{
   // This was needed as viewer was in EveWindowManager hierarchy, too.
   // el->DecParentIgnoreCnt();
   // for (auto &c: fChildren)
   // {
   //    c->DecParentIgnoreCnt();
   // }

   REveElement::RemoveElementsLocal();
}

////////////////////////////////////////////////////////////////////////////////
/// Connect to TGLViewer class-signals.

void REveViewerList::Connect()
{
   // TQObject::Connect("TGLViewer", "MouseOver(TObject*,UInt_t)",
   //                   "REveViewerList", this, "OnMouseOver(TObject*,UInt_t)");

   // TQObject::Connect("TGLViewer", "ReMouseOver(TObject*,UInt_t)",
   //                   "REveViewerList", this, "OnReMouseOver(TObject*,UInt_t)");

   // TQObject::Connect("TGLViewer", "UnMouseOver(TObject*,UInt_t)",
   //                   "REveViewerList", this, "OnUnMouseOver(TObject*,UInt_t)");

   // TQObject::Connect("TGLViewer", "Clicked(TObject*,UInt_t,UInt_t)",
   //                   "REveViewerList", this, "OnClicked(TObject*,UInt_t,UInt_t)");

   // TQObject::Connect("TGLViewer", "ReClicked(TObject*,UInt_t,UInt_t)",
   //                   "REveViewerList", this, "OnReClicked(TObject*,UInt_t,UInt_t)");

   // TQObject::Connect("TGLViewer", "UnClicked(TObject*,UInt_t,UInt_t)",
   //                   "REveViewerList", this, "OnUnClicked(TObject*,UInt_t,UInt_t)");
}

////////////////////////////////////////////////////////////////////////////////
/// Disconnect from TGLViewer class-signals.

void REveViewerList::Disconnect()
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

void REveViewerList::RepaintChangedViewers(Bool_t /*resetCameras*/, Bool_t /*dropLogicals*/)
{
   //for (auto &c: fChildren)  {
      // TGLViewer* glv = ((REveViewer*)c)->GetGLViewer();
      // if (glv->IsChanged())
      // {
      //    if (resetCameras) glv->PostSceneBuildSetup(kTRUE);
      //    if (dropLogicals) glv->SetSmartRefresh(kFALSE);

      //    glv->RequestDraw(TGLRnrCtx::kLODHigh);

      //    if (dropLogicals) glv->SetSmartRefresh(kTRUE);
      // }
   //}
}

////////////////////////////////////////////////////////////////////////////////
/// Repaint all viewers.

void REveViewerList::RepaintAllViewers(Bool_t /*resetCameras*/, Bool_t /*dropLogicals*/)
{
   // for (auto &c: fChildren) {
      // TGLViewer* glv = ((REveViewer *)c)->GetGLViewer();

      // if (resetCameras) glv->PostSceneBuildSetup(kTRUE);
      // if (dropLogicals) glv->SetSmartRefresh(kFALSE);

      // glv->RequestDraw(TGLRnrCtx::kLODHigh);

      // if (dropLogicals) glv->SetSmartRefresh(kTRUE);
   // }
}

////////////////////////////////////////////////////////////////////////////////
/// Delete annotations from all viewers.

void REveViewerList::DeleteAnnotations()
{
   // for (auto &c: fChildren) {
      // TGLViewer* glv = ((REveViewer *)c)->GetGLViewer();
      // glv->DeleteOverlayAnnotations();
  // }
}

////////////////////////////////////////////////////////////////////////////////
/// Callback done from a REveScene destructor allowing proper
/// removal of the scene from affected viewers.

void REveViewerList::SceneDestructing(REveScene* scene)
{
   for (auto &viewer: fChildren) {
      for (auto &j: viewer->RefChildren()) {
         REveSceneInfo* sinfo = (REveSceneInfo *) j;
         if (sinfo->GetScene() == scene)
            viewer->RemoveElement(sinfo);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Show / hide tooltip for various MouseOver events.
/// Must be called from slots where sender is TGLEventHandler.

void REveViewerList::HandleTooltip()
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
/// Set color brightness.

void REveViewerList::SetColorBrightness(Float_t b)
{
   REveUtil::SetColorBrightness(b, true);
}

////////////////////////////////////////////////////////////////////////////////
/// Switch background color.

void REveViewerList::SwitchColorSet()
{
   fUseLightColorSet = ! fUseLightColorSet;
   // To implement something along the lines of:
   // BeginChanges on EveWorld; // Here or in the calling function
   // for (auto &c: fChildren) {
      // REveViewer* v = (REveViewer *)c;
      // if ( fUseLightColorSet)
      //    c->UseLightColorSet();
      // else
      //    c->UseDarkColorSet();
   // }
   // EndChanges on EveWorld;
}
