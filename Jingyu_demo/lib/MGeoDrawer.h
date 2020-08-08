#include "TGeoManager.h"
#include "TEveManager.h"
#include "TFile.h"
#include "TEveGeoShape.h"
#include "TEveGeoShapeExtract.h"
#include <string>
#include <iostream>

using namespace std;

void DrawGDML(string filePath){

    TGeoManager::Import(filePath.c_str());
    gGeoManager->GetTopVolume()->Draw();

}

void DrawExtracted(string filePath){
    
    TEveManager::Create();

    TFile* file = new TFile(filePath.c_str(),"read");

    if(!file->IsOpen()){
        cout<<"file "<<filePath<<" not found"<<endl;
        return;
    }

    TEveGeoShapeExtract* shapeExtract = (TEveGeoShapeExtract*) file->Get("MuGridShape");

    TEveGeoShape* shape = TEveGeoShape::ImportShapeExtract(shapeExtract,0);

    gEve->AddGlobalElement(shape);

    shape->SetMainAlpha(0.1);

    gEve->Redraw3D();
}