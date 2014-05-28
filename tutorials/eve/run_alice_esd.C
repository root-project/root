// Macro used to prepare the environment before running the alice_esd.C or 
// alice_esd_split.C macros (when the split argument is true)

void run_alice_esd(Bool_t split = kFALSE)
{
   const char* esd_file_name = "http://root.cern.ch/files/alice_ESDs.root";
   TFile::SetCacheFileDir(".");
   TString lib(Form("aliesd/aliesd.%s", gSystem->GetSoExt()));

   if (gSystem->AccessPathName(lib, kReadPermission)) {
      TFile* f = TFile::Open(esd_file_name, "CACHEREAD");
      if (f == 0) return;
      TTree *tree = (TTree*) f->Get("esdTree");
      tree->SetBranchStatus ("ESDfriend*", 1);
      f->MakeProject("aliesd", "*", "++");
      f->Close();
      delete f;
   }
   gSystem->Load(lib);
   gROOT->ProcessLine("#define __RUN_ALICE_ESD__ 1");
   if (split) {
      gROOT->LoadMacro("SplitGLView.C+");
      gInterpreter->ExecuteMacro("alice_esd_split.C");
   }
   else {
      gROOT->LoadMacro("MultiView.C+");
      gInterpreter->ExecuteMacro("alice_esd.C");
   }
   gROOT->ProcessLine("#undef __RUN_ALICE_ESD__");
}


