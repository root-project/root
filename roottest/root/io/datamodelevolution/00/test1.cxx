#include "TFile.h"
#include "TTree.h"
#include "TBufferFile.h"
#include "TClassRef.h"

#include "DataModelV1.h"


   // Let's now inject artificially the TStreamerElement.
   // This is temporary code that will eventually be part
   // of the StreamerInfo build

class TVirtualObject;

#include "TStreamerElement.h"
#include "TStreamerInfo.h"

void test1() {

   cout << "Create an object\n";
   ACache a(6,7);
   a.CreateObjs();
   a.Print();
   
   TClassRef cl = TClass::GetClass("ACache");
 
   TBufferFile b(TBuffer::kWrite);
   b.SetWriteMode();

   cout << "Stream out object\n";
   cl->Streamer(&a, b);

   cout << "Modify object\n";
   a.GetZ();
   a.Print();

   b.Reset();
   b.SetReadMode();

   cout << "Stream in object into existing object\n";
   cl->Streamer(&a, b);

   a.Print();

   cout << "Create embedded object\n";
   Container c(8);
   c.a.CreateObjs();
   c.a.Print();

   //cl->GetStreamerInfo()->ls();

   cout << "Store on file\n";

   TFile *f = new TFile("test1.root","RECREATE");
   f->WriteObject(&a,"obj");
   f->WriteObject(&c,"cont");
   TTree *t = new TTree("tree","tree");
   t->Branch("obj",&a);
   t->Fill();
   t->Write();
   delete f;
}   
