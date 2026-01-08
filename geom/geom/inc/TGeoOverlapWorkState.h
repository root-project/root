// @(#)root/geom:$Id$
// Author: Andrei Gheata   05/01/26

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoOverlapWorkState
#define ROOT_TGeoOverlapWorkState

#include "TBuffer3D.h"
#include "TString.h"
#include "TGeoMatrix.h"
class TGeoShape;
class TGeoVolume;

struct TGeoOverlapWorkState {
   TBuffer3D fBuff1; ///< Buffer containing mesh vertices for first volume
   TBuffer3D fBuff2; ///< Buffer containing mesh vertices for second volume

   Int_t fNMeshPoints{0};                 ///< Number of mesh points allocated in buffers
   Int_t fNumPoints1{0};                  ///< Number of valid points in buffer 1
   Int_t fNumPoints2{0};                  ///< Number of valid points in buffer 2
   const TGeoShape *fLastShape1{nullptr}; ///< Last shape used to fill buffer 1
   const TGeoShape *fLastShape2{nullptr}; ///< Last shape used to fill buffer 2

   TGeoOverlapWorkState() = delete;

   explicit TGeoOverlapWorkState(Int_t nMeshPoints)
      : fBuff1(0, nMeshPoints, 3 * nMeshPoints, 0, 0, 0, 0),
        fBuff2(0, nMeshPoints, 3 * nMeshPoints, 0, 0, 0, 0),
        fNMeshPoints(nMeshPoints)
   {
   }

   void Reset()
   {
      fLastShape1 = nullptr;
      fLastShape2 = nullptr;
      fNumPoints1 = 0;
      fNumPoints2 = 0;
   }
};

// Lightweight description of a single overlap/extrusion check to perform
struct TGeoOverlapCandidate {
   TString fName; ///< display name
   TGeoVolume *fVol1 = nullptr; ///< first volume
   TGeoVolume *fVol2 = nullptr; ///< second volume
   TGeoHMatrix fMat1;         ///< matrix for first volume
   TGeoHMatrix fMat2;         ///< matrix for second volume
   Bool_t fIsOverlap = kTRUE; ///< kTRUE=overlap, kFALSE=extrusion
   Double_t fOvlp = 0.0;      ///< threshold for "illegal"
};

// Output of the numerical check (no ROOT object allocation inside)
struct TGeoOverlapResult {
   TString fName; ///< display name
   TGeoVolume *fVol1 = nullptr; ///< first volume
   TGeoVolume *fVol2 = nullptr; ///< second volume
   TGeoHMatrix fMat1; /// matrix for first volume
   TGeoHMatrix fMat2; /// matrix for second volume
   Bool_t fIsOverlap = kTRUE; ///< kTRUE=overlap, kFALSE=extrusion
   Double_t fMaxOverlap = 0.0;                   ///< overlap distance found
   std::vector<std::array<Double_t, 3>> fPoints; ///< up to N points (e.g. 100)
};

#endif // ROOT_TGeoOverlapWorkState
