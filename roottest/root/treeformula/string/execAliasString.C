{
   bool error = false;
   auto tree = new TTree ("tree", "tree");
   tree->ReadFile ("test.txt","",' ');
   tree->SetAlias("first", "(name==\"16A\")");
   tree->SetAlias("second", "count==2");
   tree->Scan();
   
   Long64_t res = tree->Draw("name", "first");
   if (res != 1) {
      error = true;
      fprintf(stdout,"Error: Draw(\"name\", \"first\") returned %lld instead of 1\n",res);
   }
   res = tree->Draw("name", "second");
   if (res != 1) {
      error = true;
      fprintf(stdout,"Error: Draw(\"name\", \"second\") returned %lld instead of 1\n",res);
   }
   res = tree->Draw("name", "first");
   if (res != 1) {
      error = true;
      fprintf(stdout,"Error: Second run of Draw(\"name\", \"first\") returned %lld instead of 1\n",res);
   }
   res = tree->Draw("name", "(name==\"16A\")");
   if (res != 1) {
      error = true;
      fprintf(stdout,"Error: Draw(\"name\", \"(name==\\\"16A\\\")\") returned %lld instead of 1\n",res);
   }
   tree->Scan("first?name:\"not first\" ");
   tree->Scan("first?name:\"not first\" ","!first");
   tree->Scan("name:count:first:second");
   tree->Scan("name:count:first:second","first");
   tree->Scan("name:count:first:second","second");
   
   // Quiet down ACLiC.
   gErrorIgnoreLevel = kWarning;
   tree->Draw("scriptAliasString.C+");
   
#ifdef ClingWorkAroundBrokenUnnamedReturn
   bool ret = error;
#else
   return error;
#endif
}

