{
  TString cmd( gSystem->GetMakeSharedLib() );
  cmd.ReplaceAll("$ObjectFiles","$ObjectFiles -L$ROOTSYS/lib -lPhysics -lMatrix");
  gSystem->SetMakeSharedLib(cmd);
  gROOT->ProcessLine(".L linktest.C+");
  
}
