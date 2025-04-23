{
   bool error = false;
   auto tree = new TTree ("tree", "tree");
   tree->ReadFile ("test.txt","",' ');

   tree->MakeProxy("generatedSel","scriptAliasString.C");

#ifdef ClingWorkAroundBrokenUnnamedReturn
   bool ret = error;
#else
   return error;
#endif
}
