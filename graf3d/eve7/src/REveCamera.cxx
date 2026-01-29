// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007, 2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveCamera.hxx>
#include <ROOT/REveManager.hxx>

#include <nlohmann/json.hpp>

using namespace ROOT::Experimental;

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

REveCamera::REveCamera() : REveElement("REveCamera")
{
   Setup(kCameraPerspXOZ, "PerspXOZ", REveVector(-1.0, 0.0, 0.0), REveVector(0.0, 1.0, 0.0));
   fCamBase.UnitTrans();
   fCamTrans.UnitTrans();
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with name

REveCamera::REveCamera(const std::string &name) : REveElement(name)
{
   Setup(kCameraPerspXOZ, name, REveVector(-1.0, 0.0, 0.0), REveVector(0.0, 1.0, 0.0));
   fCamBase.UnitTrans();
   fCamTrans.UnitTrans();
}

////////////////////////////////////////////////////////////////////////////////
/// Setup camera with type, name, direction and up vectors

void REveCamera::Setup(ECameraType type, const std::string &name, const REveVector &v1, const REveVector &v2)
{
   fType = type;
   fName = name;
   // fV1 = v1;
   // fV2 = v2;
   
   // Set up base camera matrix from direction and up vectors
   fCamBase.UnitTrans();
   fCamTrans.UnitTrans();
   
   // Create a coordinate system from v1 (direction) and v2 (up)
   REveVector dir = v1;
   dir.Normalize();
   
   REveVector up = v2;
   up.Normalize();
   
   // Right vector = dir × up
   REveVector right;
   right.fX = dir.fY * up.fZ - dir.fZ * up.fY;
   right.fY = dir.fZ * up.fX - dir.fX * up.fZ;
   right.fZ = dir.fX * up.fY - dir.fY * up.fX;
   right.Normalize();
   
   // Recalculate up = right × dir for orthogonality
   REveVector newUp;
   newUp.fX = right.fY * dir.fZ - right.fZ * dir.fY;
   newUp.fY = right.fZ * dir.fX - right.fX * dir.fZ;
   newUp.fZ = right.fX * dir.fY - right.fY * dir.fX;
   
   // Set rotation part of matrix (as row vectors)
   Double_t *M = fCamBase.Array();
   M[0] = right.fX; M[4] = right.fY; M[8]  = right.fZ;
   M[1] = newUp.fX; M[5] = newUp.fY; M[9]  = newUp.fZ;
   M[2] = dir.fX;   M[6] = dir.fY;   M[10] = dir.fZ;
   
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Set camera base matrix from array (called from client via MIR)

void REveCamera::SetCamBaseMtx(const std::vector<Double_t> &arr)
{
   if (arr.size() == 16) {
      fCamBase.SetFromArray(arr.data());
      StampObjProps();
   }
}

void REveCamera::SetCamBaseMtx(const std::string &json_str)
{
   auto j = nlohmann::json::parse(json_str);
   std::vector<Double_t> arr = j.get<std::vector<Double_t>>();
   SetCamBaseMtx(arr);
}

////////////////////////////////////////////////////////////////////////////////
/// Write core JSON for camera

Int_t REveCamera::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   Int_t ret = REveElement::WriteCoreJson(j, rnr_offset);

   j["fType"] = fType;
   j["fName"] = fName;
   // j["fV1"] = {fV1.fX, fV1.fY, fV1.fZ};
   // j["fV2"] = {fV2.fX, fV2.fY, fV2.fZ};
   
   // Stream both matrices
   // Client will read these as fMatrix arrays (16 elements each)
   const Double_t *camBaseArr = fCamBase.Array();
   j["camBase"] = std::vector<Double_t>(camBaseArr, camBaseArr + 16);

   const Double_t *camTransArr = fCamTrans.Array();
   j["camTrans"] = std::vector<Double_t>(camTransArr, camTransArr + 16);

   return ret;
}

ClassImp(REveCamera);