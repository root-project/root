{
   bool error = false;
#ifdef ClingWorkAroundMissingImplicitAuto
   TTree *tree = new TTree ("tree", "tree");
#else
   tree = new TTree ("tree", "tree");
#endif
   tree->ReadFile ("test.txt","",' ');

   tree->MakeProxy("generatedSel","scriptAliasString.C");

#ifdef ClingWorkAroundBrokenUnnamedReturn
   bool ret = error;
#else
   return error;
#endif
}
