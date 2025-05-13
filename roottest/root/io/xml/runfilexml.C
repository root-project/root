#include "test_classes.h"

void runfilexml()
{
   gSystem->Load("libXmlTestDictionaries");

   TFile *f = TFile::Open("file.xml", "recreate");
   if (!f) {
      std::cout << "Cannot create file.xml" << std::endl;
      return;
   }

   TXmlEx1* ex1 = new TXmlEx1;
   TXmlEx2* ex2 = new TXmlEx2;
   TXmlEx3* ex3 = new TXmlEx3;
   TXmlEx4* ex4 = new TXmlEx4(true);
   TXmlEx5* ex5 = new TXmlEx5(true);
   TXmlEx6* ex6 = new TXmlEx6(true);
   TXmlEx7* ex7 = new TXmlEx7(true);
   TXmlEx8* ex8 = new TXmlEx8(true);

   TH1I* h1 = new TH1I("histo1","histo title", 100, -10., 10.);
   h1->FillRandom("gaus",10000);

   TList* arr = new TList;
   for(Int_t n=0;n<10;n++) {
      TBox* b = new TBox(n*10,n*100,n*20,n*200);
      arr->Add(b, Form("option_%d_option",n));
   }

   TClonesArray* clones = new TClonesArray("TBox",10);
   for(int n=0;n<10;n++)
       new ((*clones)[n]) TBox(n*10,n*100,n*20,n*200);

   std::cout << "Writing objects to file " << std::endl;

   f->WriteObject(ex1, "ex1");
   f->WriteObject(ex2, "ex2");
   f->WriteObject(ex3, "ex3");
   f->WriteObject(ex4, "ex4");
   f->WriteObject(ex5, "ex5");
   f->WriteObject(ex6, "ex6");
   f->WriteObject(ex7, "ex7");
   f->WriteObject(ex8, "ex8");
   h1->Write("histo");
   h1->SetDirectory(0);
   arr->Write("arr",TObject::kSingleKey);
   clones->Write("clones",TObject::kSingleKey);
   delete f; f = nullptr;

   delete ex1; ex1 = nullptr;
   delete ex2; ex2 = nullptr;
   delete ex3; ex3 = nullptr;
   delete ex4; ex4 = nullptr;
   delete ex5; ex5 = nullptr;
   delete ex6; ex6 = nullptr;
   delete ex7; ex7 = nullptr;
   delete ex8; ex8 = nullptr;
   delete h1;  h1 = nullptr;
   delete arr; arr = nullptr;
   delete clones; clones = nullptr;

   f = TFile::Open("file.xml");
   if (!f) {
      std::cout << "Cannot open file.xml" << std::endl;
      return;
   }

   f->GetObject("ex1", ex1);
   f->GetObject("ex2", ex2);
   f->GetObject("ex3", ex3);
   f->GetObject("ex4", ex4);
   f->GetObject("ex5", ex5);
   f->GetObject("ex6", ex6);
   f->GetObject("ex7", ex7);
   f->GetObject("ex8", ex8);
   f->GetObject("histo", h1);
   f->GetObject("arr", arr);
   f->GetObject("clones", clones);


   std::cout << "ex1 = " << (ex1 ? "Ok" : "Error") << std::endl;
   std::cout << "ex2 = " << (ex2 ? "Ok" : "Error") << std::endl;
   std::cout << "ex3 = " << (ex3 ? "Ok" : "Error") << std::endl;
   std::cout << "ex4 = " << (ex4 ? "Ok" : "Error") << std::endl;
   std::cout << "ex5 = " << (ex5 ? "Ok" : "Error") << std::endl;
   std::cout << "ex6 = " << (ex6 ? "Ok" : "Error") << std::endl;
   std::cout << "ex7 = " << (ex7 ? "Ok" : "Error") << std::endl;
   std::cout << "ex8 = " << (ex8 ? "Ok" : "Error") << std::endl;
   std::cout << "h1 = " << (h1 ? "Ok" : "Error") << std::endl;
   std::cout << "arr = " << (arr ? "Ok" : "Error") << std::endl;
   std::cout << "clones = " << (clones ? "Ok" : "Error") << std::endl;

   delete f;
}
