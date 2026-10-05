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
#include <ROOT/REveCamera.hxx>
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
   REveElement(n, t),
   fCamera(nullptr)
{
   SetCameraType(REveCamera::kCameraPerspXOZ);
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
/// Fix the volume the 3D axis spans, as {min, max} per coordinate, instead of
/// following the scene contents. Only the axis and the clip box use it; camera
/// framing uses the content.

// clang-format off
void REveViewer::SetAxesBBox(Float_t xmin, Float_t ymin, Float_t zmin,
                             Float_t xmax, Float_t ymax, Float_t zmax)
{
   fAxesBBox[0] = xmin; fAxesBBox[1] = ymin; fAxesBBox[2] = zmin;
   fAxesBBox[3] = xmax; fAxesBBox[4] = ymax; fAxesBBox[5] = zmax;
   // clang-format on
   fHasAxesBBox = kTRUE;
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Let the 3D axis follow the scene bounding box again.

void REveViewer::ClearAxesBBox()
{
   fHasAxesBBox = kFALSE;
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Cap the redraw rate of this viewer, in frames per second; 0 is uncapped.

void REveViewer::SetRenderMaxHz(Float_t hz)
{
   fRenderMaxHz = hz < 0.f ? 0.f : (hz > 240.f ? 240.f : hz);
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Cap the rate at which this viewer applies streamed motion; 0 freezes it.

void REveViewer::SetMotionMaxHz(Float_t hz)
{
   fMotionMaxHz = hz < 0.f ? 0.f : (hz > 240.f ? 240.f : hz);
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Set the point the camera looks at and orbits around, either the world
/// origin or the center of the scene bounding box. Clients reset their camera
/// when this changes.

void REveViewer::SetCameraCenter(ECameraCenter c)
{
   fCameraCenter = c;
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Set the up axis, 0/1/2 for x/y/z; any other value means none.

void REveViewer::SetAxesUpAxis(int a)
{
   fAxesUpAxis = (a >= 0 && a <= 2) ? a : -1;
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Enable or disable client-side extrapolation of streamed motion.

void REveViewer::SetExtrapolateMotion(bool x)
{
   fExtrapolateMotion = x;
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Distance attenuation of the 3D axis labels. 0 keeps a constant pixel size
/// and 1 scales them like geometry. The clamp is [-4, 8] because inside [0, 1]
/// the effect is hard to see on a scene viewed from outside. Below 0 the far
/// labels are the larger ones.

void REveViewer::SetAxesAtten(Float_t a)
{
   fAxesAtten = a < -4.f ? -4.f : (a > 8.f ? 8.f : a);
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Label size for the 3D axis, as a fraction of viewport height, clamped to
/// [0.004, 0.05]. Above 0.05 the labels of a box axis start to meet.

void REveViewer::SetAxesFontSize(Float_t s)
{
   fAxesFontSize = s < 0.004f ? 0.004f : (s > 0.05f ? 0.05f : s);
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Label size for the hover tooltip, as a fraction of viewport height. Same
/// range as the axis labels; see SetAxesFontSize.

void REveViewer::SetTooltipFontSize(Float_t s)
{
   fTooltipFontSize = s < 0.004f ? 0.004f : (s > 0.05f ? 0.05f : s);
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Opacity of the plate behind the tooltip and kept annotations. 0 leaves the
/// text floating on the scene, 1 hides whatever is behind it.

void REveViewer::SetTooltipAlpha(Float_t a)
{
   fTooltipAlpha = a < 0.f ? 0.f : (a > 1.f ? 1.f : a);
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
/// Multiplier on the client's light intensities, clamped at zero from below.

void REveViewer::SetLightScale(Float_t s)
{
   fLightScale = s < 0.f ? 0.f : s;
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Tone curve, as EToneMapMode.

void REveViewer::SetToneMapMode(Int_t m)
{
   fToneMapMode = m;
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Knee position for kToneKnee: colours below this pass through exactly, so
/// raising it buys fidelity and spends highlight gradient.

void REveViewer::SetToneMapKnee(Float_t k)
{
   fToneMapKnee = k < 0.f ? 0.f : (k > 0.99f ? 0.99f : k);
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Ask the clients to pick a light scale that keeps the brightest channel just
/// below white. Only a client can measure that, so it reports its choice back
/// through SetLightScale().

void REveViewer::AutoTuneLights()
{
   ++fAutoTuneSerial;
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Stream Camera Info.
/// Virtual from REveElement.
int REveViewer::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   // fCamera.WriteCoreJson(j, rnr_offset);

   j["Mandatory"] = fMandatory;
   j["AxesType"] = fAxesType;
   j["CameraCenter"] = fCameraCenter;
   j["ExtrapolateMotion"] = fExtrapolateMotion;
   j["AxesUpAxis"] = fAxesUpAxis;
   j["MotionMaxHz"] = fMotionMaxHz;
   j["RenderMaxHz"] = fRenderMaxHz;
   // clang-format off
   if (fHasAxesBBox)
      j["AxesBBox"] = {fAxesBBox[0], fAxesBBox[1], fAxesBBox[2],
                       fAxesBBox[3], fAxesBBox[4], fAxesBBox[5]};
   else
      j["AxesBBox"] = nullptr;
   // clang-format on
   j["AxesAtten"] = fAxesAtten;
   j["AxesFontSize"] = fAxesFontSize;
   j["TooltipFontSize"] = fTooltipFontSize;
   j["TooltipAlpha"] = fTooltipAlpha;
   j["BlackBg"] = fBlackBackground;
   j["LightScale"] = fLightScale;
   j["ToneMapMode"] = fToneMapMode;
   j["ToneMapKnee"] = fToneMapKnee;
   j["AutoTuneSerial"] = fAutoTuneSerial;
   j["fCameraId"] = fCamera ? fCamera->GetElementId() : 0;
   j["fSyncCam"] = fSyncCamera;

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
/// Set camera reference

void REveViewer::SetCamera(::ROOT::Experimental::REveCamera *cam)
{
   fCamera = cam;
   StampObjProps();
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

void REveViewer::SetCameraByElementId(ElementId_t cameraId)
{
   auto element = gEve->FindElementById(cameraId);
   auto cam = dynamic_cast<REveCamera *>(element);

   if (cam) {
      fCamera = cam;
      StampObjProps();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set camera by type (backward compatibility with old API)

void REveViewer::SetCameraType(REveCamera::ECameraType type)
{
   for (auto &cam : fCameraList) {
      if (cam->GetType() == type) {
         fCamera = cam;
      }
   }

   fCamera = CreateCamera(type);
   fCameraList.push_back(fCamera);
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
// Create Camera Based on Type Enum (Lines 161-190)
REveCamera *REveViewer::CreateCamera(ECameraType type)
{
   REveCamera *cam = nullptr;

   struct CameraDef {
      REveCamera::ECameraType type;
      const char *name;
      REveVector v1;
      REveVector v2;
   };

   static const CameraDef predefinedCameras[] = {
      // Perspective cameras
      {REveCamera::kCameraPerspXOZ, "PerspXOZ", REveVector(-1.0, 0.0, 0.0), REveVector(0.0, 1.0, 0.0)},
      {REveCamera::kCameraPerspYOZ, "PerspYOZ", REveVector(0.0, -1.0, 0.0), REveVector(1.0, 0.0, 0.0)},
      {REveCamera::kCameraPerspXOY, "PerspXOY", REveVector(-1.0, 0.0, 0.0), REveVector(0.0, 0.0, 1.0)},
      // Orthographic cameras
      {REveCamera::kCameraOrthoXOY, "OrthoXOY", REveVector(0.0, 0.0, 1.0), REveVector(0.0, 1.0, 0.0)},
      {REveCamera::kCameraOrthoXOZ, "OrthoXOZ", REveVector(0.0, -1.0, 0.0), REveVector(0.0, 0.0, 1.0)},
      {REveCamera::kCameraOrthoZOY, "OrthoZOY", REveVector(-1.0, 0.0, 0.0), REveVector(0.0, 1.0, 0.0)},
      {REveCamera::kCameraOrthoZOX, "OrthoZOX", REveVector(0.0, -1.0, 0.0), REveVector(1.0, 0.0, 0.0)},
      // Orthographic negative camera
      {REveCamera::kCameraOrthoXnOY, "OrthoXnOY", REveVector(0.0, 0.0, -1.0), REveVector(0.0, 1.0, 0.0)},
      {REveCamera::kCameraOrthoXnOZ, "OrthoXnOZ", REveVector(0.0, 1.0, 0.0), REveVector(0.0, 0.0, 1.0)},
      {REveCamera::kCameraOrthoZnOY, "OrthoZnOY", REveVector(1.0, 0.0, 0.0), REveVector(0.0, 1.0, 0.0)},
      {REveCamera::kCameraOrthoZnOX, "OrthoZnOX", REveVector(0.0, 1.0, 0.0), REveVector(1.0, 0.0, 0.0)}};

   // Create and add all predefined cameras
   for (const auto &camDef : predefinedCameras) {
      if (type == camDef.type) {
         cam = new REveCamera(camDef.name);
         gEve->GetCameras()->AddElement(cam);
         cam->Setup(camDef.type, camDef.name, camDef.v1, camDef.v2);
      }
   }

   return cam;
}
