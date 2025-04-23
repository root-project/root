void writefile(int version = 1) {
   gROOT->ProcessLine(Form(".L data%d.C+",version));

#ifdef ClingWorkAroundAutoParseTooPrecise
   const char *versions[] = { "ZERO","ONE","TWO","THREE","FOUR","FIVE","SIX" };
   gROOT->ProcessLine(Form("#define VERSION_%s",versions[version]));
#endif
#ifdef ClingWorkAroundAutoParseDeclaration
   gInterpreter->AutoParse("data");
#endif

   TFile *f = new TFile(Form("data%d.root",version),"RECREATE");
#if defined(ClingWorkAroundMissingDynamicScope)
#if _MSC_VER
#define Ox "0x"
#else
#define Ox ""
#endif
   gROOT->ProcessLine(TString::Format("TFile *f = (TFile*)%s%p;\n", Ox, f));
   gROOT->ProcessLine("::data *a = new ::data; f->WriteObject(a,\"myobj\");");
   gROOT->ProcessLine("Tdata *b = new Tdata;  f->WriteObject(b,\"myTobj\");");
#else
   ::data *a = new ::data;
   Tdata *b = new Tdata;
   f->WriteObject(a,"myobj");
   f->WriteObject(b,"myTobj");
#endif
   f->Write();
   delete f;
}
