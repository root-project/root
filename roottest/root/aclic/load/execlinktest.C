{
   return 0;
#if 0
   TString cmd( gSystem->GetMakeSharedLib() );
   if (strcmp("win32",gSystem->GetBuildArch())==0) {
      cmd.ReplaceAll("$LinkedLibs","$LinkedLibs -LIBPATH:%ROOTSYS\\lib libPhysics.lib libMatrix.lib libRIO.lib");
   } else {
      cmd.ReplaceAll("$ObjectFiles","$ObjectFiles -L$ROOTSYS/lib -lPhysics -lMatrix -lRIO");
   }
   gSystem->SetMakeSharedLib(cmd);
   // Commenting out these two lines prevents the crash
   gErrorIgnoreLevel = kWarning;
   gROOT->ProcessLine(".L linktest.C+"); 
#endif
}
