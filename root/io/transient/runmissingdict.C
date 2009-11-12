{
   gSystem->Load("libCintex"); ROOT::Cintex::Cintex::Enable();
   gROOT->ProcessLine(".L missingdict_rflx.cpp+");
   TClass::GetClass("Content")->GetStreamerInfo();
   return 0;
}
