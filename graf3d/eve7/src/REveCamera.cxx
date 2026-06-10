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
#include <iostream>

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

void REveCamera::Setup(ECameraType type, const std::string &name, const REveVector &hAxis, const REveVector &vAxis)
{
   fType = type;
   fName = name;

   // Set up base camera matrix from direction and up vectors
   fCamBase.UnitTrans();
   fCamTrans.UnitTrans();

   fCamBase.SetBaseVec(1, hAxis.fX, hAxis.fY, hAxis.fZ);
	fCamBase.SetBaseVec(3, vAxis.fX, vAxis.fY, vAxis.fZ);

   REveVector y = vAxis.Cross(hAxis);
   fCamBase.SetBaseVec(2, y.fX, y.fY, y.fZ);

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

void REveCamera::SetCamTransMtx(const std::vector<Double_t> &arr)
{
   if (arr.size() == 16) {
      fCamTrans.SetFromArray(arr.data());
      fInitialized = kTRUE;
   }
}

// Set translation matrix with an array of 17 floats
// The first 16 floats are 4x4 matrix element
// The 17th value is setting the zoom value in orthographic type
void REveCamera::SetCamTransMtxStr(const char *ins)
{
   std::stringstream ss(ins);
   std::vector<double> arr;
   std::string item;
   while (std::getline(ss, item, ',')) {
      arr.push_back(std::stod(item));
   }

   fOrthoZoom = arr.back();
   arr.pop_back();
   fCamTrans.SetFromArray(arr.data());

   fInitialized = true;

   StampObjProps();
   SetCamTransMtx(arr);
}

void REveCamera::SetOrthoZoom(float zoom)
{
   fOrthoZoom = zoom;
}

////////////////////////////////////////////////////////////////////////////////
/// Write core JSON for camera

Int_t REveCamera::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   Int_t ret = REveElement::WriteCoreJson(j, rnr_offset);

   j["fType"] = fType;
   j["fName"] = fName;
   j["fInitialized"] = fInitialized;  // Stream to client

   // Stream both matrices
   // Client will read these as fMatrix arrays (16 elements each)
   const Double_t *camBaseArr = fCamBase.Array();
   j["camBase"] = std::vector<Double_t>(camBaseArr, camBaseArr + 16);

   const Double_t *camTransArr = fCamTrans.Array();
   j["camTrans"] = std::vector<Double_t>(camTransArr, camTransArr + 16);
   j["fZoom"] = fOrthoZoom;

   return ret;
}
