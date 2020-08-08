#include "TGeoManager.h"
#include "TFile.h"
#include "TEveManager.h"
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TEveGeoNode.h"
#include "TEveProjections.h"
#include "TEveProjectionManager.h"
#include "TEveStraightLineSet.h"
#include "TEvePointSet.h"
#include "TEveWindowManager.h"
#include "TEveProjectionAxes.h"
#include "TEveScene.h"
#include "TEveWindow.h"
#include "TEveBox.h"
#include "TRandom.h"
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include "TROOT.h"

//experimental
#include "../lib/MGex.h"

//#define USE_STANDARD_MULTIVIEW 1
class TEveManager;

using namespace std;

TEveGeoShape* topShape;

TEveProjectionManager *fRPhiMgr;
TEveProjectionManager *fRhoZMgr;
 
TEveViewer            *f3DView;
TEveViewer            *fRPhiView;
TEveViewer            *fRhoZView;
 
TEveScene             *fRPhiGeomScene;
TEveScene             *fRhoZGeomScene;
TEveScene             *fRPhiEventScene;
TEveScene             *fRhoZEventScene;

vector<TVector3> pmtPos;

void InitializeViewer();
void HandleEventDisplay();
void InitializeVolumeColor();
void ExportPMPos(string path);
void VolumeTraversal(TGeoVolume* vol);

void EveDisplay(){

    topShape = GetShapeFromGDML("../models/MuGrid_3.gdml");

    gEve->AddGlobalElement(topShape);

    InitializeViewer();

    InitializeVolumeColor();

    //HandleEventDisplay();

    gEve->Redraw3D(kTRUE);


}

void InitializeViewer(){
    // 1. Generate scenes and projections.
    // 2. Create multi-view
    // 2. Add geometry elements.

    // Constructor --- creates required scenes, projection managers
      // and GL viewers.
 
      // Scenes
      //========
 
      fRPhiGeomScene  = gEve->SpawnNewScene("RPhi Geometry",
                                            "Scene holding projected geometry for the RPhi view.");
      fRhoZGeomScene  = gEve->SpawnNewScene("RhoZ Geometry",
                                            "Scene holding projected geometry for the RhoZ view.");
      fRPhiEventScene = gEve->SpawnNewScene("RPhi Event Data",
                                            "Scene holding projected event-data for the RPhi view.");
      fRhoZEventScene = gEve->SpawnNewScene("RhoZ Event Data",
                                            "Scene holding projected event-data for the RhoZ view.");
 
 
      // Projection managers
      //=====================

      fRPhiMgr = new TEveProjectionManager(TEveProjection::kPT_RPhi);
      gEve->AddToListTree(fRPhiMgr, kFALSE);
      {
         TEveProjectionAxes* a = new TEveProjectionAxes(fRPhiMgr);
         a->SetMainColor(kWhite);
         a->SetTitle("R-Phi");
         a->SetTitleSize(0.05);
         a->SetTitleFont(102);
         a->SetLabelSize(0.025);
         a->SetLabelFont(102);
         fRPhiGeomScene->AddElement(a);
      }

      fRhoZMgr = new TEveProjectionManager(TEveProjection::kPT_XZ);
      gEve->AddToListTree(fRhoZMgr, kFALSE);
      {
         TEveProjectionAxes* a = new TEveProjectionAxes(fRhoZMgr);
         a->SetMainColor(kWhite);
         a->SetTitle("XZ");
         a->SetTitleSize(0.05);
         a->SetTitleFont(102);
         a->SetLabelSize(0.025);
         a->SetLabelFont(102);
         fRhoZGeomScene->AddElement(a);
      }

    fRPhiMgr->ImportElements(topShape,fRPhiGeomScene);

    fRhoZMgr->ImportElements(topShape,fRhoZGeomScene);
 
 
      // Viewers
      //=========


 
      TEveWindowSlot *slot = 0;
      TEveWindowPack *pack = 0;
        
      slot = TEveWindow::CreateWindowInTab(gEve->GetBrowser()->GetTabRight());
      pack = slot->MakePack();
      pack->SetElementName("Multi View");
      pack->SetHorizontal();
      pack->SetShowTitleBar(kTRUE);
      pack->NewSlot()->MakeCurrent();
      f3DView = gEve->SpawnNewViewer("3D View", "");
      f3DView->AddScene(gEve->GetGlobalScene());
      f3DView->AddScene(gEve->GetEventScene());
 
      pack = pack->NewSlot()->MakePack();
      pack->SetShowTitleBar(kFALSE);
      pack->NewSlot()->MakeCurrent();
      fRPhiView = gEve->SpawnNewViewer("RPhi View", "");
      fRPhiView->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
      fRPhiView->AddScene(fRPhiGeomScene);
      fRPhiView->AddScene(fRPhiEventScene);
 
      pack->NewSlot()->MakeCurrent();
      fRhoZView = gEve->SpawnNewViewer("XZ View", "");
      fRhoZView->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
      fRhoZView->AddScene(fRhoZGeomScene);
      fRhoZView->AddScene(fRhoZEventScene);

   

}


void HandleEventDisplay(){

    TEveStraightLineSet* lineSet = new TEveStraightLineSet;
    TEvePointSet* pointSet = new TEvePointSet;
    gEve->AddGlobalElement(lineSet);
    gEve->AddGlobalElement(pointSet);

    lineSet->AddLine(0,0,0,20,20,20);
    pointSet->SetNextPoint(3,3,3);

    lineSet->SetMainColor(kRed);
    pointSet->SetMarkerColor(kRed);
    pointSet->SetMarkerSize(2);

    


    gEve->Redraw3D();

}


void InitializeVolumeColor(){

    topShape->SetMainAlpha(0.1);
    for(auto iter = topShape->BeginChildren();iter!=topShape->EndChildren();iter++){
        TEveElement* element = *iter;
        element->SetMainAlpha(0.1);
        element->SetMainColor(kWhite);
    }
    gEve->Redraw3D();
}

//Traversal through the geometry tree in GeoManager and calculate PM position
void ExportPMPos(string path){
    if(gGeoManager==NULL){
        cout<<"gGeoManager is NULL"<<endl;
        return;
    }

    pmtPos.clear();

}

//recursive
void VolumeTraversal(TGeoVolume* vol){
    if(vol==NULL)return;
    

}