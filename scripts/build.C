void build(const char *filename,const char *lib = 0, const char *obj = 0) {

   if (obj!=0 && strlen(obj) ) {
      TString s = gSystem->GetMakeSharedLib();
      TString r(" $ObjectFiles ");
      r.Append(obj);
      s.ReplaceAll(" $ObjectFiles",r);
      //gDebug = 5;
      gSystem->SetMakeSharedLib(s);
   }
   if (lib && strlen(lib)) {
      TString liblist(lib);
      TObjArray *libs = liblist.Tokenize(" ");
      TIter iter(libs);
      TObjString *objstr;
      while ( (objstr=(TObjString*)iter.Next()) ) {
         gSystem->Load(objstr->String());
      }
   }
   int result = gSystem->CompileMacro(filename,"kc");
   if (!result) gApplication->Terminate(1);
}

