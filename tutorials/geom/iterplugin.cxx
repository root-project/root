#include "TROOT.h"
#include "TGeoNode.h"
#include "TGeoVolume.h"
#include "TGeoManager.h"
#include "TVirtualGeoPainter.h"

//_______________________________________________________________________________
class iterplugin : public TGeoIteratorPlugin
{
public:
   iterplugin() : TGeoIteratorPlugin(), fColor(kGreen), fReplica(1) {}
   virtual ~iterplugin() {}
   // Process current node
   virtual void ProcessNode();
   void         Select(Int_t replica, Int_t color) {fReplica=replica; fColor=color;}

   Int_t fColor;             // Current color
   Int_t fReplica;           // replica number (1 to 4)

   ClassDef(iterplugin, 0)   // A simple user iterator plugin that changes volume color
};

ClassImp(iterplugin)

void iterplugin::ProcessNode()
{
   if (!fIterator) return;
   TString path;
   fIterator->GetPath(path);
   if (!path.Contains(Form("REPLICA_%d",fReplica))) return;
   Int_t level = fIterator->GetLevel();
   TGeoVolume *vol = fIterator->GetNode(level)->GetVolume();
   vol->SetLineColor(fColor);
}

