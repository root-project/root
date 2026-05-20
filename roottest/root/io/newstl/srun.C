enum EWhich {
   kVector = 1,
   kDeque = 2,
   kList = 4,
   kSet = 8,
   kMultiSet = 16,
   kMap = 32,
   kMultiMap = 64,
   kRVec = 128,
   kEnd = 256
};

void srun(const TString &srcpath = "", int which = kEnd - 1, int exec = 2) {

   // use current directory as build directory
   gSystem->SetBuildDir(gSystem->pwd(), true);

   int result = gSystem->CompileMacro(srcpath + "TestHelpers.C","k");
   if (result != 0 && result != 1) return;

   TString arg = exec > 1 ? "(true)" : "(false)";

   if ( which & kRVec ) {

      result = gSystem->CompileMacro(srcpath + "rvecHolder.C","k");
      if (result != 0 && result != 1) return;

      result = gSystem->CompileMacro(srcpath + "rtest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("rtest" + arg);

   }

   if ( which & kVector ) {

      result = gSystem->CompileMacro(srcpath + "vectorHolder.C","k");
      if (result != 0 && result != 1) return;

      result = gSystem->CompileMacro(srcpath + "vtest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("vtest" + arg);

   }

   if ( which & kDeque ) {

      result = gSystem->CompileMacro(srcpath + "dequeHolder.C","k");
      if (result != 0 && result != 1) return;

      result = gSystem->CompileMacro(srcpath + "dtest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("dtest" + arg);

   }

   if ( which & kList ) {

      result = gSystem->CompileMacro(srcpath + "listHolder.C","k");
      if (result != 0 && result != 1) return;

      result = gSystem->CompileMacro(srcpath + "ltest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("ltest" + arg);

   }

   if ( which & kSet ) {

      result = gSystem->CompileMacro(srcpath + "setHolder.C","k");
      if (result != 0 && result != 1) return;

      result = gSystem->CompileMacro(srcpath + "stest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("stest" + arg);

   }

   if ( which & kMultiSet ) {

      result = gSystem->CompileMacro(srcpath + "multisetHolder.C","k");
      if (result != 0 && result != 1) return;

      result = gSystem->CompileMacro(srcpath + "ttest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("ttest" + arg);

   }

   if ( which & kMap ) {

      result = gSystem->CompileMacro(srcpath + "mapHolder.C","k");
      if (result != 0 && result != 1) return;

      result = gSystem->CompileMacro(srcpath + "mtest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("mtest" + arg);

   }

   if ( which & kMultiMap ) {

      result = gSystem->CompileMacro(srcpath + "multimapHolder.C","k");
      if (result != 0 && result != 1) return;

      result = gSystem->CompileMacro(srcpath + "ntest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("ntest" + arg);

   }

   // pass++;
}
