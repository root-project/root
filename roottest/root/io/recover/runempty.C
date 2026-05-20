void func(const char * name);

void printmap() 
{
   if (gDebug) {
      TFile *h = new TFile("temp.root");
      h->Map();
      delete h;
   }
}

void runempty(bool working = false){

   // delete old file, if any

   if (!working) {
      TFile *f = new TFile("temp.root","RECREATE");
      f->Write();
      f->Close();
      delete f;
      f = 0;
      printmap();
   }

   //fill file with dummy trees

   printf("\n");
   func("tree1");
   printmap();
   printf("\n");
   func("tree2");
   printmap();

   TFile *g = new TFile("temp.root");

}

void func(const char * name) {

   TFile *g = new TFile("temp.root","UPDATE");

   // delete any pre-existing trees of the same name

   char nameCycle[100];
   snprintf(nameCycle,100,"%s;*",name);
   g->Delete(nameCycle);

   TTree *tree1 = new TTree(name,name);

   Float_t flt = 1;
   tree1->Branch("branch",&flt,"temp/F");

   for (int i = 0; i < 500; i++) {
      tree1->Fill();
   }

   g->Write();
   tree1->Reset();
   delete tree1;
   tree1 = 0;

   g->Close();
   delete g;
   g = 0;

}
