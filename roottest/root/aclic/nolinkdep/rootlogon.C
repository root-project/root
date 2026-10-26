{
   TString cmd = gSystem->GetMakeSharedLib();
#if defined(G__APPLE)
   //fprintf(stdout,"Fixing for MACOS\n");
   cmd.ReplaceAll("$LinkedLibs","-Wl,-dead_strip_dylibs $LinkedLibs");
#elif defined(G__GNUC)
   //fprintf(stdout,"Fixing for MACOS\n");
   cmd.ReplaceAll("$LinkedLibs","-Wl,--as-needed $LinkedLibs");
#endif
   gSystem->SetMakeSharedLib(cmd);
}
