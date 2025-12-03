#include "test_classes.h"

void runRootClasses()
{
   gSystem->Load("libJsonTestClasses");


   TH1I *h1 = new TH1I("histo1","histo title", 100, -10., 10.);
   for (Int_t bin=1;bin<=100;++bin)
      h1->SetBinContent(bin, bin % 12);
   h1->ResetBit(kMustCleanup); // reset bit while it is always done in TH1::Streamer()

   TObject *obj = new TObject();

   TBox *box = new TBox(11,22,33,44);

   TList *arr = new TList;
   for(Int_t n=0;n<10;n++) {
      TBox* b = new TBox(n*10,n*100,n*20,n*200);
      arr->Add(b, Form("option_%d_option",n));
   }

   TClonesArray *clones = new TClonesArray("TBox",10);
   for(int n=0;n<10;n++)
       new ((*clones)[n]) TBox(n*10,n*100,n*20,n*200);

   TMap *map = new TMap;
   for (int n=0;n<10;n++) {
      TObjString* str = new TObjString(Form("Str%d",n));
      TNamed* nnn = new TNamed(Form("Name%d",n), Form("Title%d",n));
      map->Add(str,nnn);
   }

   TJsonEx14 *ex14 = new TJsonEx14; ex14->Init();

   TString json;

   std::cout << " ====== TObject representation ===== " << std::endl;
   json = TBufferJSON::ToJSON(obj);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== TH1I representation ===== " << std::endl;
   json = TBufferJSON::ToJSON(h1);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== TBox representation ===== " << std::endl;
   json = TBufferJSON::ToJSON(box);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== TList representation ===== " << std::endl;
   json = TBufferJSON::ToJSON(arr);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== TClonesArray representation ===== " << std::endl;
   json = TBufferJSON::ToJSON(clones);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== TMap representation ===== " << std::endl;
   json = TBufferJSON::ToJSON(map);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;
   std::cout << " ====== TJsonEx14 with different ROOT collections ===== " << std::endl;
   json = TBufferJSON::ToJSON(ex14);
   testJsonReading(json);
   std::cout << json << std::endl << std::endl;

   delete obj;
   delete h1;
   delete box;
   delete arr;
   delete clones;
   delete map;
   delete ex14;
}
