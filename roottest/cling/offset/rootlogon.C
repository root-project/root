{
   TString cmd( gSystem->GetMakeSharedLib() );
   cmd.ReplaceAll("-Werror","");
   gSystem->SetMakeSharedLib(cmd);
}
