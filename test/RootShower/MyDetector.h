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
#include "TGeometry.h"
#include "TBRIK.h"
#include "TNode.h"
#include "TMaterial.h"

class MyDetector : public TObject {

private:

    Int_t       fMatter;           // Matter identification
    Char_t      fMaterialName[40]; // Material name
    Double_t    fMinX,fMinY,fMinZ; // Detector min. boundaries
    Double_t    fMaxX,fMaxY,fMaxZ; // Detector max. boundaries
    Double_t    fDimX,fDimY,fDimZ; // Detector dimensions
    Double_t    fI;                // Ionisation constant for current material (in MeV/cm)
    Double_t    fPreconst;         // First factor in the Bethe-Bloch equation for current material (in MeV/cm)
    Double_t    fEc;               // Critical Energy for current material (in MeV)
    Double_t    fX0;               // Radiation Length for current material (in cm)
    Double_t    fdT;               // Time step dT for current material (in ms - about 0.015 times X_0/c)
    Double_t    fTheta0;           // Scatter angle for current material
    Double_t    fTotalELoss;       // Total Energy loss by particles into the detector
    TGeometry   *fGeometry;        // Detector geometry

public:

    MyDetector();
    virtual ~MyDetector();
    void        Init(Int_t mat,Double_t dimx,Double_t dimy,Double_t dimz);
    void        ChangeDimensions(Double_t dimx,Double_t dimy,Double_t dimz);
    void        UpdateShape();
    void        ChangeMaterial(Int_t mat);
    Int_t       GetMaterial() { return fMatter; }
    Char_t     *GetMaterialName() { return fMaterialName; }
    Double_t    GetI() { return fI; }
    Double_t    GetPreconst() { return fPreconst; }
    Double_t    GetEc() { return fEc; }
    Double_t    GetX0() { return fX0; }
    Double_t    GetdT() { return fdT; }
    Double_t    GetTheta0() { return fTheta0; }
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
    TGeometry  *GetGeometry() { return fGeometry; }

    void        SetMatter(Int_t mat) { fMatter = mat; }
    void        SetDimensions(Double_t dimx, Double_t dimy, Double_t dimz)
                { fDimX = dimx; fDimY = dimy; fDimZ = dimz; }
    void        SetI(Double_t val) { fI = val; }
    void        SetPreconst(Double_t val) { fPreconst = val; }
    void        SetEc(Double_t val) { fEc = val; }
    void        SetX0(Double_t val) { fX0 = val; }
    void        SetdT(Double_t val) { fdT = val; }
    void        SetTheta0(Double_t val) { fTheta0 = val; }
    void        AddELoss(Double_t val) { fTotalELoss += val; }

    ClassDef(MyDetector,1)   // Detector structure

};

#endif // MYDETECTOR_H

