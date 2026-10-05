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
#include <ROOT/REveCamera.hxx> 

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
   // set alias instead
   using ECameraType = REveCamera::ECameraType;
   
   // backward compatibility
   static constexpr ECameraType kCameraPerspXOZ   = REveCamera::kCameraPerspXOZ;
   static constexpr ECameraType kCameraPerspYOZ   = REveCamera::kCameraPerspYOZ;
   static constexpr ECameraType kCameraPerspXOY   = REveCamera::kCameraPerspXOY;
   static constexpr ECameraType kCameraOrthoXOY   = REveCamera::kCameraOrthoXOY;
   static constexpr ECameraType kCameraOrthoXOZ   = REveCamera::kCameraOrthoXOZ;
   static constexpr ECameraType kCameraOrthoZOY   = REveCamera::kCameraOrthoZOY;
   static constexpr ECameraType kCameraOrthoZOX   = REveCamera::kCameraOrthoZOX;
   static constexpr ECameraType kCameraOrthoXnOY  = REveCamera::kCameraOrthoXnOY;
   static constexpr ECameraType kCameraOrthoXnOZ  = REveCamera::kCameraOrthoXnOZ;
   static constexpr ECameraType kCameraOrthoZnOY  = REveCamera::kCameraOrthoZnOY;
   static constexpr ECameraType kCameraOrthoZnOX  = REveCamera::kCameraOrthoZnOX;

   enum EAxesType {
      kAxesNone,
      kAxesOrigin,
      kAxesEdge
   };

   /// Point the camera looks at and orbits around after a camera reset.
   enum ECameraCenter {
      kCameraCenterOrigin, ///< world origin, (0, 0, 0)
      kCameraCenterBBox    ///< center of the scene bounding box
   };

   /// Tone curve applied to the rendered buffer. Values match the shader.
   // clang-format off
   enum EToneMapMode {
      kToneReinhard = 0,
      kToneExposure = 1,
      kToneLinear   = 2,  ///< no curve; exact colours, but anything over 1 clips
      kToneKnee     = 3   ///< identity below the knee, smooth roll-off above
   };
   // clang-format on

private:
   REveViewer(const REveViewer&) = delete;
   REveViewer& operator=(const REveViewer&) = delete;

   REveCamera* fCamera{0};

   EAxesType fAxesType{kAxesNone};

   ECameraCenter fCameraCenter{kCameraCenterOrigin};

   /// Let the client extrapolate streamed motion between updates. Off holds
   /// each object where the last update put it, which shows the raw update rate.
   Bool_t fExtrapolateMotion{kTRUE};

   /// Which axis points up: 0/1/2 for x/y/z, -1 (default) for none. When set,
   /// the box axis draws the floor panel along it instead of the far face.
   Int_t fAxesUpAxis{-1};

   // clang-format off
   Bool_t  fHasAxesBBox{kFALSE};  ///< see SetAxesBBox()
   Float_t fAxesBBox[6]{};        ///< xmin, ymin, zmin, xmax, ymax, zmax

   Float_t fMotionMaxHz{60.f};    ///< see SetMotionMaxHz(); 0 freezes
   Float_t fRenderMaxHz{0.f};     ///< see SetRenderMaxHz(); 0 is uncapped
   bool      fBlackBackground{false};

   /// Distance attenuation of the 3D axis labels; see SetAxesAtten().
   Float_t   fAxesAtten{1.f};

   /// 3D axis label size, as a fraction of viewport height. Baked into the
   /// glyph geometry, so a change costs a rebuild on the client.
   Float_t   fAxesFontSize{0.015f};

   /// Tooltip text size, same units as fAxesFontSize.
   Float_t   fTooltipFontSize{0.012f};

   /// Opacity of the plate behind the tooltip and kept annotations, in [0, 1].
   /// Unlike the font size this does reach annotations already placed.
   Float_t   fTooltipAlpha{0.85f};

   /// Multiplier on the client's light intensities; see SetLightScale().
   Float_t   fLightScale{0.85};
   Int_t     fToneMapMode{kToneKnee};
   Float_t   fToneMapKnee{0.95};
   /// Bumped by AutoTuneLights(); the client re-measures when it changes.
   Int_t     fAutoTuneSerial{0};
   // clang-format on

   bool fMandatory{true};
   std::string fPostStreamFlag;

   std::vector<REveCamera*> fCameraList;

   ROOT::Experimental::REveCamera* CreateCamera(ECameraType type);

   bool fSyncCamera{true};

public:
   REveViewer(const std::string &n="REveViewer", const std::string &t="");
   ~REveViewer() override;

   void Redraw(Bool_t resetCameras=kFALSE);

   virtual void AddScene(REveScene* scene);
   // XXX Missing RemoveScene() ????

   // Camera setters
   void SetCamera(ROOT::Experimental::REveCamera *cam);
   REveCamera* GetCamera() const { return fCamera;}
   void SetCameraByElementId(ElementId_t cameraId); // set camera via ElementID
   void SetCameraType(REveCamera::ECameraType type);

   void SyncCamera(bool s) {fSyncCamera = s;}
   bool GetSyncCamera() const {return fSyncCamera;}

   // The getters return the value last sent to the clients, not what a client
   // is showing.
   EAxesType GetAxesType() const { return fAxesType; }
   void SetAxesType(int);

   ECameraCenter GetCameraCenter() const { return fCameraCenter; }
   void SetCameraCenter(ECameraCenter c);

   // clang-format off
   Bool_t GetExtrapolateMotion() const { return fExtrapolateMotion; }
   void   SetExtrapolateMotion(bool);

   Int_t GetAxesUpAxis() const { return fAxesUpAxis; }
   void  SetAxesUpAxis(int);

   void SetAxesBBox(Float_t xmin, Float_t ymin, Float_t zmin,
                    Float_t xmax, Float_t ymax, Float_t zmax);
   void ClearAxesBBox();
   Bool_t HasAxesBBox() const { return fHasAxesBBox; }

   /// Cap on how often this viewer applies streamed motion, in updates per
   /// second, clamped to [0, 240]. Zero freezes motion; SetExtrapolateMotion()
   /// only controls drawing between updates.
   Float_t GetMotionMaxHz() const { return fMotionMaxHz; }
   void    SetMotionMaxHz(Float_t);

   /// Cap on how often the animation loop redraws this viewer, in frames per
   /// second, clamped to [0, 240]. Zero means uncapped, unlike SetMotionMaxHz().
   Float_t GetRenderMaxHz() const { return fRenderMaxHz; }
   void    SetRenderMaxHz(Float_t);
   // clang-format on

   bool GetBlackBackground() const { return fBlackBackground; }
   void SetBlackBackground(bool);

   Float_t GetAxesAtten() const { return fAxesAtten; }
   void SetAxesAtten(Float_t a);

   Float_t GetAxesFontSize() const { return fAxesFontSize; }
   void SetAxesFontSize(Float_t s);

   Float_t GetTooltipFontSize() const { return fTooltipFontSize; }
   void SetTooltipFontSize(Float_t s);

   Float_t GetTooltipAlpha() const { return fTooltipAlpha; }
   void SetTooltipAlpha(Float_t a);

   Float_t GetLightScale() const { return fLightScale; }
   void SetLightScale(Float_t s);

   Int_t GetToneMapMode() const { return fToneMapMode; }
   void SetToneMapMode(Int_t m);

   Float_t GetToneMapKnee() const { return fToneMapKnee; }
   void SetToneMapKnee(Float_t k);

   void AutoTuneLights();

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
