{
// Fill out the code of the actual test
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   if (1) {
#endif
#ifndef SECOND_RUN
   gROOT->ProcessLine(".L test_classes.h+");
#endif

#if defined(ClingWorkAroundMissingDynamicScope) && !defined(SECOND_RUN)
#define SECOND_RUN
   gROOT->ProcessLine(".x runfilexml.C");
#else
   TFile *f = TFile::Open("file.xml", "recreate");
   if (f==0) {
      cout << "Cannot create file.xml" << endl;   
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
   
   cout << "Writing objects to file " << endl;
   
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
   delete f; f = 0;
   
   delete ex1; ex1 = 0;
   delete ex2; ex2 = 0;
   delete ex3; ex3 = 0;
   delete ex4; ex4 = 0;
   delete ex5; ex5 = 0;
   delete ex6; ex6 = 0;
   delete ex7; ex7 = 0;
   delete ex8; ex8 = 0;
   delete h1;  h1 = 0;
   delete arr; arr = 0;
   delete clones; clones = 0;
   
#ifdef ClingReinstateRedeclarationAllowed
   TFile *f = TFile::Open("file.xml");
#else
   f = TFile::Open("file.xml");
#endif
   if (f==0) {
      cout << "Cannot open file.xml" << endl;   
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


   cout << "ex1 = " << (ex1 ? "Ok" : "Error") << endl;
   cout << "ex2 = " << (ex2 ? "Ok" : "Error") << endl;
   cout << "ex3 = " << (ex3 ? "Ok" : "Error") << endl;
   cout << "ex4 = " << (ex4 ? "Ok" : "Error") << endl;
   cout << "ex5 = " << (ex5 ? "Ok" : "Error") << endl;
   cout << "ex6 = " << (ex6 ? "Ok" : "Error") << endl;
   cout << "ex7 = " << (ex7 ? "Ok" : "Error") << endl;
   cout << "ex8 = " << (ex8 ? "Ok" : "Error") << endl;
   cout << "h1 = " << (h1 ? "Ok" : "Error") << endl;
   cout << "arr = " << (arr ? "Ok" : "Error") << endl;
   cout << "clones = " << (clones ? "Ok" : "Error") << endl;

   delete f;
#endif
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   }
#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
