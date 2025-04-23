#ifdef ClingWorkAroundMissingDynamicScope
void ReadOldObject(int /* version */ = 2){
#else
void ReadOldObject(int version = 2){
   if (version == 1) {
      gROOT->LoadMacro("MyClass.cxx+");
   } else {
      gROOT->LoadMacro("MyClassOld.cxx+");      
   }
#endif
   TFile* f = new TFile("oldArrayObject.root","READ");
	MyClass* my = (MyClass*)f->Get("MyClass");
	Int_t* a = my->GetArray();
	// fprintf(stdout,"array=%p\n",a);
   for (Int_t i = 0; i<5; i++){
		printf("array[%d] = %d\n",i,a[i]);
	}
	f->Close();
	return;
} 
