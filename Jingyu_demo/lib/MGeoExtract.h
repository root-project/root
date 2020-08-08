#include "TEveManager.h"
#include "TEveGeoShape.h"
#include "TEveGeoShapeExtract.h"
#include "TEveGeoNode.h"
#include "TString.h"
#include "TFile.h"
#include "TKey.h"
#include "TGeoManager.h"
#include "TEveElement.h"
#include "TEveTrans.h"
#include <string>
using namespace std;

TEveGeoShape* GetShapeFromNode(TGeoNode* node);



TEveGeoShape* GetShapeFromGDML(string filePath){

    TGeoManager::Import((filePath).c_str());

    if(!gEve)TEveManager::Create();

    TEveGeoShape* topShape = new TEveGeoShape("top","top");

    topShape = GetShapeFromNode(gGeoManager->GetTopNode());

    return topShape;

}

//Recursive funcion which will traversal the TGeo tree.
TEveGeoShape* GetShapeFromNode(TGeoNode* node){
    
    if(node == NULL)return NULL;

    //operation
    TEveGeoShape* shape = new TEveGeoShape(node->GetName(),node->GetName());
    shape->SetPickable(kTRUE);
    shape->RefMainTrans().SetFrom(*node->GetMatrix());
    shape->SetShape((TGeoShape*)node->GetVolume()->GetShape()->Clone());
    

    //look in daughters
    int n = node->GetVolume()->GetNdaughters();
    for(int i=0;i<n;i++){
        auto nextNode = node->GetVolume()->GetNode(i);
        auto childShape = GetShapeFromNode(nextNode);
        if(childShape)shape->AddElement(childShape);
    }

    return shape;

}



