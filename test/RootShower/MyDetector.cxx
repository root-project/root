// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

#include <stdlib.h>
#include <TROOT.h>

#ifndef MYDETECTOR_H
#include "MyDetector.h"
#endif

//______________________________________________________________________________
//
// MyDetector class implementation
//______________________________________________________________________________


ClassImp(MyDetector)

//______________________________________________________________________________
MyDetector::MyDetector() 
{
    // Standard constructor
}

//______________________________________________________________________________
MyDetector::~MyDetector() 
{
    // Destructor
}

//______________________________________________________________________________
void MyDetector::Init(Int_t mat,Double_t dimx,Double_t dimy,Double_t dimz)
{
    //
    // Initialize detector with material and dimensions
    //
    fTotalELoss = 0.0;
    fDimX = dimx;
    fDimY = dimy;
    fDimZ = dimz;
    fMaxX = fDimX / 2.;
    fMinX = -fMaxX;
    fMaxY = fDimY / 2.;
    fMinY = -fMaxY;
    fMaxZ = fDimZ / 2.;
    fMinZ = -fMaxZ;
    fGeometry = new TGeometry("detector","detector");
    fGeometry->cd();
    switch(mat) {
        case Fe:
            sprintf(fMaterialName,"Iron (Fe)");
            break;
        case Pb:
            sprintf(fMaterialName,"Lead (Pb)");
            break;
        case Polystyrene:
            sprintf(fMaterialName,"Polystyrene scintillator");
            break;
        case BGO:
            sprintf(fMaterialName,"Bismuth germanate (BGO)");
            break;
        case CsI:
            sprintf(fMaterialName,"Cesium iodide (CsI)");
            break;
        case NaI:
            sprintf(fMaterialName,"Sodium iodide (NaI)");
            break;
        default:
            sprintf(fMaterialName,"Polystyrene scintillator");
            break;
    }
    ChangeMaterial(mat);
}

//______________________________________________________________________________
void MyDetector::ChangeDimensions(Double_t dimx,Double_t dimy,Double_t dimz)
{
    //
    // Change detector dimensions
    //
    fDimX = dimx;
    fDimY = dimy;
    fDimZ = dimz;
    fMaxX = fDimX / 2.;
    fMinX = -fMaxX;
    fMaxY = fDimY / 2.;
    fMinY = -fMaxY;
    fMaxZ = fDimZ / 2.;
    fMinZ = -fMaxZ;
    UpdateShape();
}

//______________________________________________________________________________
void MyDetector::UpdateShape()
{
    //
    // Update geometry shape (after change of detector dimensions)
    //
    TBRIK       *shape;
    TNode       *node;
    fGeometry->cd();
    shape = new TBRIK("shape","shape",fMaterialName,fDimX/2.0,fDimY/2.0,fDimZ/2.0);
    shape->SetLineColor(7);
    node = new TNode("node","node","shape");

}

//______________________________________________________________________________
void MyDetector::ChangeMaterial(Int_t mat)
{
    //
    // Change current detector's material and update related physical properties
    //
    TMaterial   *fMaterial;
    Double_t     x;
    fMatter = mat;
    fGeometry->cd();
    switch(fMatter) {
//          TMaterial(name, title, A, Z, density, radl, interl);
        case Fe:
            sprintf(fMaterialName,"Iron");
            fMaterial = new TMaterial(fMaterialName,fMaterialName,55.85f,26,7.87f,1.76f,131.9f);
            break;
        case Pb:
            sprintf(fMaterialName,"Lead (Pb)");
            fMaterial = new TMaterial(fMaterialName,fMaterialName,207.2f,82,11.35f,0.56f,194.0f);
            break;
        case Polystyrene:
            sprintf(fMaterialName,"Polystyrene scintillator");
            fMaterial = new TMaterial(fMaterialName,fMaterialName,13.01f,7,1.032f,42.4f,81.9f);
            break;
        case BGO:
            sprintf(fMaterialName,"Bismuth germanate (BGO)");
            fMaterial = new TMaterial(fMaterialName,fMaterialName,175.92f,74,7.1f,1.12f,157.0f);
            break;
        case CsI:
            sprintf(fMaterialName,"Cesium iodide (CsI)");
            fMaterial = new TMaterial(fMaterialName,fMaterialName,129.90f,54,4.53f,1.85f,167.0f);
            break;
        case NaI:
            sprintf(fMaterialName,"Sodium iodide (NaI)");
            fMaterial = new TMaterial(fMaterialName,fMaterialName,117.10f,50,3.67f,2.59f,151.0f);
            break;
        default:
            sprintf(fMaterialName,"Polystyrene scintillator");
            fMaterial = new TMaterial(fMaterialName,fMaterialName,13.01f,7,1.032f,42.4f,81.9f);
            break;
    }
    UpdateShape();

    // Ionisation constant in MeV/cm
    fI = 16.0e-06 * (TMath::Power(fMaterial->GetZ(),0.9));
    fI *= 1.0e-03; // in GeV/cm...

    // first factor in the Bethe-Bloch equation in MeV/cm
    fPreconst = 0.3071 * fMaterial->GetDensity() * (fMaterial->GetZ() / fMaterial->GetA());
    fPreconst *= 1.0e-03; // in GeV/cm...
    
    // Critical Energy in MeV
    fEc = 800.0 / (fMaterial->GetZ() + 1.2);
    fEc *= 1.0e-03; // in GeV...

    // Radiation Length in cm
    fX0 = fMaterial->GetRadLength();

    // Time step dT in ms about 0.015 times X_0/c
    fdT = 0.015 * (fX0 / C);

    // Scatter angle 
    x = fdT * C;
    fTheta0 = TMath::Sqrt(2.0) *13.6 * TMath::Sqrt(x / fX0) *(1 + 0.038 * TMath::Log(x / fX0));
    fTheta0 *= 1.0e-03; 

}













