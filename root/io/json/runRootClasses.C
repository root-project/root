{
   TH1I* h1 = new TH1I("histo1","histo title", 100, -10., 10.);
   for (Int_t bin=1;bin<=100;++bin)
      h1->SetBinContent(bin, bin % 12);

   TObject* obj = new TObject();

   TBox* box = new TBox(11,22,33,44);

   TList* arr = new TList;
   for(Int_t n=0;n<10;n++) {
      TBox* b = new TBox(n*10,n*100,n*20,n*200);
      arr->Add(b, Form("option_%d_option",n));
   }

   TClonesArray* clones = new TClonesArray("TBox",10);
   for(int n=0;n<10;n++)
       new ((*clones)[n]) TBox(n*10,n*100,n*20,n*200);

   TMap* map = new TMap;
   for (int n=0;n<10;n++) {
      TObjString* str = new TObjString(Form("Str%d",n));
      TNamed* nnn = new TNamed(Form("Name%d",n), Form("Title%d",n));
      map->Add(str,nnn);
   }


   cout << " ====== TObject representation ===== " << endl;
   cout << TBufferJSON::ToJSON(obj) << endl << endl;
   cout << " ====== TH1I representation ===== " << endl;
   cout << TBufferJSON::ToJSON(h1) << endl << endl;
   cout << " ====== TBox representation ===== " << endl;
   cout << TBufferJSON::ToJSON(box) << endl << endl;
   cout << " ====== TList representation ===== " << endl;
   cout << TBufferJSON::ToJSON(arr) << endl << endl;
   cout << " ====== TClonesArray representation ===== " << endl;
   cout << TBufferJSON::ToJSON(clones) << endl << endl;
   cout << " ====== TMap representation ===== " << endl;
   cout << TBufferJSON::ToJSON(map) << endl << endl;

#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
