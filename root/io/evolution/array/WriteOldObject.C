void WriteOldObject(){
#ifndef ClingWorkAroundMissingDynamicScope
	gROOT->LoadMacro("MyClass.cxx+");
#endif
	TFile* f = new TFile("oldArrayObject.root","RECREATE");
	MyClass* my = new MyClass();
	Int_t a[5] = {10,20,30,40,50};
	my->SetArray(a);
	my->Write();
	f->Close();
	return;
} 
