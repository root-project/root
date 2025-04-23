// void run(Int_t opt=0) {
{

   int result = gSystem->CompileMacro("TestHelpers.C","k");
   if (result != 0 && result != 1) return;

   if ( which & kRVec ) {

      result = gSystem->CompileMacro("rvecHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("rtest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("rtest(exec-1)");

   }

   if ( which & kVector ) {

      result = gSystem->CompileMacro("vectorHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("vtest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("vtest(exec-1)");

   }

   if ( which & kDeque ) {

      result = gSystem->CompileMacro("dequeHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("dtest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("dtest(exec-1)");

   }

   if ( which & kList ) {

      result = gSystem->CompileMacro("listHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("ltest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("ltest(exec-1)");

   }

   if ( which & kSet ) {

      result = gSystem->CompileMacro("setHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("stest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("stest(exec-1)");

   }

   if ( which & kMultiSet ) {

      result = gSystem->CompileMacro("multisetHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("ttest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("ttest(exec-1)");

   }

   if ( which & kMap ) {

      result = gSystem->CompileMacro("mapHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("mtest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("mtest(exec-1)");

   }

   if ( which & kMultiMap ) {

      result = gSystem->CompileMacro("multimapHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("ntest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) gROOT->ProcessLine("ntest(exec-1)");

   }

   pass++;
}
