// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MyDetector                                                           //
// defines a simple detector class with one geometry, one material      //
// and the physical properties of the detector's material               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef MYDETECTOR_H
#define MYDETECTOR_H

#include "constants.h"

#include "TObject.h"
#include "TObjArray.h"
#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMedium.h"
#include "TGeoVolume.h"
#include "TGeoMatrix.h"

class MyDetector : public TObject {

private:
    Double_t        fMinX,fMinY,fMinZ; // Detector min. boundaries
    Double_t        fMaxX,fMaxY,fMaxZ; // Detector max. boundaries
    Double_t        fDimX,fDimY,fDimZ; // Detector dimensions
    Double_t        fI[6];             // Ionisation constant for current material (in MeV/cm)
    Double_t        fPreconst[6];      // First factor in the Bethe-Bloch equation for current material (in MeV/cm)
    Double_t        fEc[6];            // Critical Energy for current material (in MeV)
    Double_t        fX0[6];            // Radiation Length for current material (in cm)
    Double_t        fdT[6];            // Time step dT for current material (in ms - about 0.015 times X_0/c)
    Double_t        fTheta0[6];        // Scatter angle for current material
    Double_t        fTotalELoss;       // Total Energy loss by particles into the detector

public:
    MyDetector();
    virtual ~MyDetector();
    void        Init();
    Double_t    GetI(Int_t idx) { return fI[idx]; }
    Double_t    GetPreconst(Int_t idx) { return fPreconst[idx]; }
    Double_t    GetEc(Int_t idx) { return fEc[idx]; }
    Double_t    GetX0(Int_t idx) { return fX0[idx]; }
    Double_t    GetdT(Int_t idx) { return fdT[idx]; }
    Double_t    GetTheta0(Int_t idx) { return fTheta0[idx]; }
    Double_t    GetDimX() { return fDimX; }
    Double_t    GetDimY() { return fDimY; }
    Double_t    GetDimZ() { return fDimZ; }
    Double_t    GetMaxX() { return fMaxX; }
    Double_t    GetMaxY() { return fMaxY; }
    Double_t    GetMaxZ() { return fMaxZ; }
    Double_t    GetMinX() { return fMinX; }
    Double_t    GetMinY() { return fMinY; }
    Double_t    GetMinZ() { return fMinZ; }
    Double_t    GetTotalELoss() { return fTotalELoss; }
    void        GetDimensions(Double_t *dimx, Double_t *dimy, Double_t *dimz)
                { *dimx = fDimX; *dimy = fDimY; *dimz = fDimZ; }
    TGeoManager *GetGeoManager() { return gGeoManager; }

    void        SetI(Int_t idx, Double_t val) { fI[idx] = val; }
    void        SetPreconst(Int_t idx, Double_t val) { fPreconst[idx] = val; }
    void        SetEc(Int_t idx, Double_t val) { fEc[idx] = val; }
    void        SetX0(Int_t idx, Double_t val) { fX0[idx] = val; }
    void        SetdT(Int_t idx, Double_t val) { fdT[idx] = val; }
    void        SetTheta0(Int_t idx, Double_t val) { fTheta0[idx] = val; }
    void        AddELoss(Double_t val) { fTotalELoss += val; }
    void        ClearELoss() { fTotalELoss = 0.0; }

    ClassDef(MyDetector,1)   // Detector structure
};

#endif // MYDETECTOR_H

