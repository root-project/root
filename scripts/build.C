void build(const char *filename,const char *lib = 0, int i=0) {

   if (i==1) {
      if (filename==0 || lib==0) return;
      TString s = gSystem->GetMakeSharedLib();
      TString r(" $ObjectFiles ");
      r.Append(lib);
      s.ReplaceAll(" $ObjectFiles",r);
      //gDebug = 5;
      gSystem->SetMakeSharedLib(s);
   } else if (i==0) {
      if (lib) gSystem->Load(lib);
   }
   int result = gSystem->CompileMacro(filename,"kc");
   if (!result) gApplication->Terminate(1);
}

