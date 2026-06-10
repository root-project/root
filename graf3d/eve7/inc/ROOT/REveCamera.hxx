// @(#)root/eve7:$Id$
// Authors: Yuxiao Wang, 2025

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveCamera
#define ROOT7_REveCamera

#include <ROOT/REveElement.hxx>
#include <ROOT/REveVector.hxx>
#include <ROOT/REveTrans.hxx>

#include <string>

namespace ROOT {
namespace Experimental {

class REveCamera : public REveElement
{
public:
   enum ECameraType {
      // Perspective
      kCameraPerspXOZ,  // XOZ floor
      kCameraPerspYOZ,  // YOZ floor
      kCameraPerspXOY,  // XOY floor
      // Orthographic
      kCameraOrthoXOY,  // Looking down Z axis, X horz, Y vert
      kCameraOrthoXOZ,  // Looking along Y axis, X horz, Z vert
      kCameraOrthoZOY,  // Looking along X axis, Z horz, Y vert
      kCameraOrthoZOX,  // Looking along Y axis, Z horz, X vert
      // Orthographic negative
      kCameraOrthoXnOY, // Looking along Z axis, -X horz, Y vert
      kCameraOrthoXnOZ, // Looking down Y axis, -X horz, Z vert
      kCameraOrthoZnOY, // Looking down X axis, -Z horz, Y vert
      kCameraOrthoZnOX  // Looking down Y axis, -Z horz, X vert
   };

private:
   ECameraType fType;
   std::string fName;
   
   // Camera transformation matrices
   REveTrans   fCamBase;   // Base camera matrix (main positioning)
   REveTrans   fCamTrans;
   Bool_t      fInitialized{kFALSE};
   float       fOrthoZoom{1.f};

public:
   REveCamera();
   REveCamera(const std::string &name);
   virtual ~REveCamera() {}

   void Setup(ECameraType type, const std::string &name, const REveVector &v1, const REveVector &v2);

   ECameraType GetType() const { return fType; }
   const std::string &GetCameraName() const { return fName; }
   
   // Camera matrix accessors
   REveTrans &RefCamBase() { return fCamBase; }
   const REveTrans &GetCamBase() const { return fCamBase; }

   REveTrans &RefCamTrans() { return fCamTrans; }
   const REveTrans &GetCamTrans() const { return fCamTrans; }
   
   void SetCamBase(const REveTrans &base) { fCamBase = base; StampObjProps(); }
   
   // receive mtx from client
   void SetCamBaseMtx(const std::vector<Double_t> &arr);
   void SetCamBaseMtx(const std::string &json_str);

   void SetCamTransMtx(const std::vector<Double_t> &arr);
   void SetCamTransMtxStr(const char* json_str);

   void SetOrthoZoom(float);

   Bool_t IsInitialized() const { return fInitialized; }
   void SetInitialized(Bool_t val) { fInitialized = val; }

   void BuildRenderData() override{};

   Int_t WriteCoreJson(nlohmann::json &j, Int_t rnr_offset) override;

   ClassDef(REveCamera, 0);
};

} // namespace Experimental
} // namespace ROOT

#endif
