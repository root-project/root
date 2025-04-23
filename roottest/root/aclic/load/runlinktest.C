{
   TString cmd( gSystem->GetMakeSharedLib() );
   cmd.ReplaceAll("$ObjectFiles","$ObjectFiles -L$ROOTSYS/lib -lPhysics -lMatrix -lRIO");
   gSystem->SetMakeSharedLib(cmd);
   gErrorIgnoreLevel = kWarning;
   gROOT->ProcessLine(".L linktest.C+");
}
