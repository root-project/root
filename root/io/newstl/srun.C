// void run(Int_t opt=0) {
{

   int result = gSystem->CompileMacro("TestHelpers.C","k");
   if (result != 0 && result != 1) return;

   if ( which & kVector ) {

      result = gSystem->CompileMacro("vectorHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("vtest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) vtest(exec-1);

   }

   if ( which & kDeque ) {

      result = gSystem->CompileMacro("dequeHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("dtest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) dtest(exec-1);

   }

   if ( which & kList ) {

      result = gSystem->CompileMacro("listHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("ltest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) ltest(exec-1);

   }

   if ( which & kSet ) {

      result = gSystem->CompileMacro("setHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("stest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) stest(exec-1);

   }

   if ( which & kMultiSet ) {

      result = gSystem->CompileMacro("multisetHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("ttest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) ttest(exec-1);

   }

   if ( which & kMap ) {

      result = gSystem->CompileMacro("mapHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("mtest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) mtest(exec-1);

   }

   if ( which & kMultiMap ) {

      result = gSystem->CompileMacro("multimapHolder.C","k");
      if (result != 0 && result != 1) return;
      
      result = gSystem->CompileMacro("ntest.C","k");
      if (result != 0 && result != 1) return;

      if (exec) ntest(exec-1);

   }

   pass++;
}
