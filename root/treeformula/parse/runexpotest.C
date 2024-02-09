{
//("sshl1e + sshl2e")
TTree *t = new TTree("T","T");
float one,two;
one = 1.5;
two = 2.0;
t->Branch("var1e",&one,"var1e/F");
t->Branch("var2e",&two,"var2e/F");
t->Branch("e3",&two,"e3/F");
t->Branch("e",&two,"e/F");
t->Branch("pp",&one,"pp/F");
t->Fill();
t->Fill();

if (TClass::GetDict("TTreeFormula")==0) gSystem->Load("libTreePlayer");

#ifdef ClingWorkAroundMissingDynamicScope
TTree *T = t;
#endif
TTreeFormula * f1= new TTreeFormula("testing","(var1e)+(var2e)",T);
f1->Print();

TTreeFormula * f2= new TTreeFormula("testing","(var1e+var2e)",T);
f2->Print();

TTreeFormula * f3= new TTreeFormula("testing","(var1e+1e+3+var2e+3e-4)",T);
f3->Print();

TTreeFormula * f3a= new TTreeFormula("testing","(var1e-3+var2e+4)",T);
f3a->Print();

TTreeFormula * f4= new TTreeFormula("testing","(3e-4)",T);
f4->Print();

TTreeFormula * f5= new TTreeFormula("testing","(var1e+1e+3+var2e-3)",T);
f5->Print();

TTreeFormula * f6= new TTreeFormula("testing","(var2e-3)",T);
f6->Print();

TTreeFormula * f7= new TTreeFormula("testing","e3",T);
f7->Print();

TTreeFormula *t1 = new TTreeFormula("t1","e*e-pp*pp",T);
t1->Print();

TTreeFormula *t2 = new TTreeFormula("t2","(e*e)-pp*pp",T);
t2->Print();

auto myhist = new TH1F("myhisto","test",100,-10,10);
T->Draw("abs(((e*e)-pp*pp)-(e*e-pp*pp))>0.0001 >> myhisto","","goff");
if (myhist->GetMean()!=0 || myhist->GetRMS()!=0) {
    printf("Error: TTreeFormula does not think that (e*e)-pp*pp)==(e*e-pp*pp) [%f,%f]\n",
           myhist->GetMean(), myhist->GetRMS());
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(1);
#else
   return 1;
#endif
}

#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
