{
   TString cmd = gSystem->GetMakeSharedLib();
#if defined(G__APPLE)
   //fprintf(stdout,"Fixing for MACOS\n");
   cmd.ReplaceAll("$DepLibs","-Wl,-dead_strip_dylibs $DepLibs");
#elif defined(G__GNUC)
   //fprintf(stdout,"Fixing for MACOS\n");
   cmd.ReplaceAll("$DepLibs","-Wl,--as-needed $DepLibs");
#endif
   gSystem->SetMakeSharedLib(cmd);
}
