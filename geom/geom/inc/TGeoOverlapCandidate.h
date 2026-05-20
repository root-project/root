// @(#)root/geom:$Id$
// Author: Andrei Gheata   05/01/26

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoOverlapCandidate
#define ROOT_TGeoOverlapCandidate

#include "TBuffer3D.h"
#include "TString.h"
#include "TGeoMatrix.h"
class TGeoShape;
class TGeoVolume;

// Lightweight description of a single overlap/extrusion check to perform
struct TGeoOverlapCandidate {
   TString fName;               ///< display name
   TGeoVolume *fVol1 = nullptr; ///< first volume
   TGeoVolume *fVol2 = nullptr; ///< second volume
   TGeoHMatrix fMat1;           ///< matrix for first volume
   TGeoHMatrix fMat2;           ///< matrix for second volume
   Bool_t fIsOverlap = kTRUE;   ///< kTRUE=overlap, kFALSE=extrusion
   Double_t fOvlp = 0.0;        ///< threshold for "illegal"
};

// Output of the numerical check (no ROOT object allocation inside)
struct TGeoOverlapResult {
   TString fName;                                ///< display name
   TGeoVolume *fVol1 = nullptr;                  ///< first volume
   TGeoVolume *fVol2 = nullptr;                  ///< second volume
   TGeoHMatrix fMat1;                            /// matrix for first volume
   TGeoHMatrix fMat2;                            /// matrix for second volume
   Bool_t fIsOverlap = kTRUE;                    ///< kTRUE=overlap, kFALSE=extrusion
   Double_t fMaxOverlap = 0.0;                   ///< overlap distance found
   std::vector<std::array<Double_t, 3>> fPoints; ///< up to N points (e.g. 100)
};

#endif // ROOT_TGeoOverlapWorkState
