{
//("sshl1e + sshl2e")
TTree *t = new TTree("T","T");
float one,two;
t->Branch("var1e",&one,"var1e/F");
t->Branch("var2e",&two,"var2e/F");

if (TClass::GetDict("TTreeFormula")==0) gSystem->Load("libTreePlayer");

TTreeFormula * f1= new TTreeFormula("testing","(var1e)+(var2e)",T);
f1->Print();

TTreeFormula * f2= new TTreeFormula("testing","(var1e+var2e)",T);
f2->Print();

TTreeFormula * f3= new TTreeFormula("testing","(var1e+1e+3+var2e+3e-4)",T);
f3->Print();

TTreeFormula * f7= new TTreeFormula("testing","(var1e-3+var2e+4)",T);
f7->Print();

TTreeFormula * f4= new TTreeFormula("testing","(3e-4)",T);
f4->Print();

TTreeFormula * f5= new TTreeFormula("testing","(var1e+1e+3+var2e-3)",T);
f5->Print();

TTreeFormula * f6= new TTreeFormula("testing","(var2e-3)",T);
f6->Print();

}
