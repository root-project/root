{
   bool error = false;
   tree = new TTree ("tree", "tree");
   tree.ReadFile ("test.txt","",' ');

   tree->MakeProxy("generatedSel","scriptAliasString.C");

   return error;
}
