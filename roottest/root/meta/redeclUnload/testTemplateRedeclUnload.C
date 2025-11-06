void testTemplateRedeclUnload()
{
   gROOT->ProcessLine("gSystem->Load(\"libTemplateRedecl_dictrflx\");");
   gROOT->ProcessLine("gInterpreter->AutoParse(\"RandomClass\");");

   // Undo the last 2 commands, which unloads the templates.
   gROOT->ProcessLine(".undo 2");

   // Before the fix: cling crashes here with:
   //   "Passed first decl twice, invalid redecl chain!"
   gROOT->ProcessLine("RandomClass obj;");
}
