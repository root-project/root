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
void MyDetector::Init()
{
    //
    // Initialize detector with material and dimensions
    //
    Int_t           i;
    Double_t        x;
    TGeoMaterial    *Material;
    TGeoMaterial    *fIron;
    TGeoMaterial    *fLead;
    TGeoMaterial    *fPolystyrene;
    TGeoMaterial    *fBGO;
    TGeoMaterial    *fCsI;
    TGeoMaterial    *fNaI;
    TGeoMedium      *fScintillator[4];
    TGeoMedium      *fDiscriminator;
    TGeoMedium      *fCalorimeter;
    TGeoVolume      *fVolume[7];
    TGeoTranslation *fTrans[6];

    fTotalELoss = 0.0;
    fDimX = 40.0;
    fDimY = 42.0;
    fDimZ = 40.0;
    fMaxX = fDimX / 2.;
    fMinX = -fMaxX;

    fMinY = 0.0;
    fMaxY = fDimY+fMinY;

    fMaxZ = fDimZ / 2.;
    fMinZ = -fMaxZ;

    new TGeoManager("MyDetector", "MyDetector");

    fTrans[0] = new TGeoTranslation(0., 2., 0.);
    fTrans[1] = new TGeoTranslation(0., 5.0, 0.);
    fTrans[2] = new TGeoTranslation(0., 8.0, 0.);
    fTrans[3] = new TGeoTranslation(0., 12.0, 0.);
    fTrans[4] = new TGeoTranslation(0., 26.0, 0.);
    fTrans[5] = new TGeoTranslation(0., 40.0, 0.);
    
    fIron = new TGeoMaterial("Iron",55.85f,26,7.87f,1.76f,131.9f);
    fLead = new TGeoMaterial("Lead",207.2f,82,11.35f,0.56f,194.0f);
    fPolystyrene = new TGeoMaterial("Polystyrene",13.01f,7,1.032f,42.4f,81.9f);
    fBGO = new TGeoMaterial("BGO",175.92f,74,7.1f,1.12f,157.0f);
    fCsI = new TGeoMaterial("CsI",129.90f,54,4.53f,1.85f,167.0f);
    fNaI = new TGeoMaterial("NaI",117.10f,50,3.67f,2.59f,151.0f);

    fScintillator[0] = new TGeoMedium("SCINT0",1, fPolystyrene);
    fDiscriminator   = new TGeoMedium("DISCR", 2, fLead);
    fScintillator[1] = new TGeoMedium("SCINT1",3, fPolystyrene);
    fScintillator[2] = new TGeoMedium("SCINT2",4, fPolystyrene);
    fCalorimeter     = new TGeoMedium("CALOR", 5, fNaI);
    fScintillator[3] = new TGeoMedium("SCINT3",6, fPolystyrene);
    
    TGeoMaterial *fMat = new TGeoMaterial("VOID");
    TGeoMedium *fMed = new TGeoMedium("MED",0,fMat);
    fVolume[0] = gGeoManager->MakeBox("TOP",fMed,40,42,40);
    fVolume[0]->SetVisibility(kFALSE);
    gGeoManager->SetTopVolume(fVolume[0]);
    
    fVolume[1] = gGeoManager->MakeBox("BOX0",fScintillator[0], 20.0,2.00,20.0);
    fVolume[1]->SetLineColor(7);
    fVolume[1]->SetTransparency(50);
    fVolume[1]->SetLineWidth(1);
    fVolume[0]->AddNode(fVolume[1],0,fTrans[0]);

    fVolume[2] = gGeoManager->MakeBox("BOX1",fDiscriminator, 20.0,1.00,20.0);
    fVolume[2]->SetLineColor(15);
    fVolume[2]->SetTransparency(50);
    fVolume[2]->SetLineWidth(1);
    fVolume[0]->AddNode(fVolume[2],1,fTrans[1]);

    fVolume[3] = gGeoManager->MakeBox("BOX2",fScintillator[1], 20.0,2.00,20.0);
    fVolume[3]->SetLineColor(7);
    fVolume[3]->SetTransparency(50);
    fVolume[3]->SetLineWidth(1);
    fVolume[0]->AddNode(fVolume[3],2,fTrans[2]);

    fVolume[4] = gGeoManager->MakeBox("BOX3",fScintillator[2], 20.0,2.00,20.0);
    fVolume[4]->SetLineColor(7);
    fVolume[4]->SetTransparency(50);
    fVolume[4]->SetLineWidth(1);
    fVolume[0]->AddNode(fVolume[4],3,fTrans[3]);

    fVolume[5] = gGeoManager->MakeBox("BOX4",fCalorimeter, 20.0,12.00,20.0);
    fVolume[5]->SetLineColor(38);
    fVolume[5]->SetTransparency(50);
    fVolume[5]->SetLineWidth(1);
    fVolume[0]->AddNode(fVolume[5],4,fTrans[4]);

    fVolume[6] = gGeoManager->MakeBox("BOX5",fScintillator[3], 20.0,2.00,20.0);
    fVolume[6]->SetLineColor(7);
    fVolume[6]->SetTransparency(50);
    fVolume[6]->SetLineWidth(1);
    fVolume[0]->AddNode(fVolume[6],5,fTrans[5]);

    gGeoManager->CloseGeometry();

    for(i=0;i<6;i++) {
        Material = fVolume[i+1]->GetMaterial();
        // Ionisation constant in MeV/cm
        fI[i] = 16.0e-06 * (TMath::Power(Material->GetZ(),0.9));
        fI[i] *= 1.0e-03; // in GeV/cm...

        // first factor in the Bethe-Bloch equation in MeV/cm
        fPreconst[i] = 0.3071 * Material->GetDensity() * 
            (Material->GetZ() / Material->GetA());
        fPreconst[i] *= 1.0e-03; // in GeV/cm...
    
        // Critical Energy in MeV
        fEc[i] = 800.0 / (Material->GetZ() + 1.2);
        fEc[i] *= 1.0e-03; // in GeV...

        // Radiation Length in cm
        fX0[i] = Material->GetRadLen();

        // Time step dT in ms about 0.015 times X_0/c
        fdT[i] = 0.015 * (fX0[i] / CSpeed);

        // Scatter angle 
        x = fdT[i] * CSpeed;
        fTheta0[i] = TMath::Sqrt(2.0) * 13.6 * TMath::Sqrt(x / fX0[i]) * 
            (1.0 + 0.038 * TMath::Log(x / fX0[i]));
        fTheta0[i] *= 1.0e-03; 
    }
    
}














