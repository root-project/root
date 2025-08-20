// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007, 2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveViewer
#define ROOT7_REveViewer

#include <ROOT/REveElement.hxx>

namespace ROOT {
namespace Experimental {

class REveScene;

////////////////////////////////////////////////////////////////////////////////
/// REveViewer
/// Reve representation of TGLViewer.
////////////////////////////////////////////////////////////////////////////////

class REveViewer : public REveElement
{
public:
   enum ECameraType
   {
      // Perspective
      kCameraPerspXOZ,  // XOZ floor
      kCameraPerspYOZ,  // YOZ floor
      kCameraPerspXOY,  // XOY floor
      // Orthographic
      kCameraOrthoXOY,  // Looking down  Z axis,  X horz, Y vert
      kCameraOrthoXOZ,  // Looking along Y axis,  X horz, Z vert
      kCameraOrthoZOY,  // Looking along X axis,  Z horz, Y vert
      kCameraOrthoZOX,  // Looking along Y axis,  Z horz, X vert
      // nOrthographic
      kCameraOrthoXnOY, // Looking along Z axis, -X horz, Y vert
      kCameraOrthoXnOZ, // Looking down  Y axis, -X horz, Z vert
      kCameraOrthoZnOY, // Looking down  X axis, -Z horz, Y vert
      kCameraOrthoZnOX  // Looking down  Y axis, -Z horz, X vert
   };

   enum EAxesType {
      kAxesNone,
      kAxesOrigin,
      kAxesEdge
   };

   // For the moment REveCamera is internal class
   class REveCamera
   {
      ECameraType fType;
      std::string fName;
      REveVector fV2;
      REveVector fV1;

      public:
       REveCamera() { Setup(kCameraPerspXOZ, "PerspXOZ", REveVector(-1.0, 0.0, 0.0), REveVector(0.0, 1.0, 0.0));}
       ~REveCamera() {}

       void Setup(ECameraType type, const std::string& name, REveVector v1, REveVector v2);

       ECameraType GetType() const { return fType; }

       int WriteCoreJson(nlohmann::json &j, Int_t /*rnr_offset*/);
   };

private:
   REveViewer(const REveViewer&) = delete;
   REveViewer& operator=(const REveViewer&) = delete;

   REveCamera fCamera;
   EAxesType fAxesType{kAxesNone};
   bool      fBlackBackground{false};

   bool fMandatory{true};
   std::string fPostStreamFlag;

public:
   REveViewer(const std::string &n="REveViewer", const std::string &t="");
   ~REveViewer() override;

   void Redraw(Bool_t resetCameras=kFALSE);

   virtual void AddScene(REveScene* scene);
   // XXX Missing RemoveScene() ????

   void SetCameraType(ECameraType t);
   ECameraType GetCameraType() const { return fCamera.GetType(); }

   void SetAxesType(int);
   void SetBlackBackground(bool);

   void DisconnectClient();
   void ConnectClient();

   void SetMandatory(bool x);
   bool GetMandatory() { return fMandatory; }

   void RemoveElementLocal(REveElement *el) override;
   void RemoveElementsLocal() override;
   Int_t WriteCoreJson(nlohmann::json &cj, Int_t rnr_offset) override;
};


////////////////////////////////////////////////////////////////////////////////
/// REveViewerList
/// List of Viewers providing common operations on REveViewer collections.
////////////////////////////////////////////////////////////////////////////////

class REveViewerList : public REveElement
{
private:
   REveViewerList(const REveViewerList&) = delete;
   REveViewerList& operator=(const REveViewerList&) = delete;

protected:
   Bool_t        fShowTooltip;

   Float_t       fBrightness;
   Bool_t        fUseLightColorSet;

   void HandleTooltip();

public:
   REveViewerList(const std::string &n="REveViewerList", const std::string &t="");
   ~REveViewerList() override;

   void AddElement(REveElement* el) override;
   void RemoveElementLocal(REveElement* el) override;
   void RemoveElementsLocal() override;

   // --------------------------------

   virtual void Connect();
   virtual void Disconnect();

   void RepaintChangedViewers(Bool_t resetCameras, Bool_t dropLogicals);
   void RepaintAllViewers(Bool_t resetCameras, Bool_t dropLogicals);
   void DeleteAnnotations();

   void SceneDestructing(REveScene* scene);

   // --------------------------------

   Bool_t  GetShowTooltip()     const { return fShowTooltip; }
   void    SetShowTooltip(Bool_t x)   { fShowTooltip = x; }

   Float_t GetColorBrightness() const { return fBrightness; }
   void    SetColorBrightness(Float_t b);

   Bool_t  UseLightColorSet()   const { return fUseLightColorSet; }
   void    SwitchColorSet();
 //  Int_t WriteCoreJson(nlohmann::json &cj, Int_t rnr_offset) override;
};

} // namespace Experimental
} // namespace ROOT

#endif
