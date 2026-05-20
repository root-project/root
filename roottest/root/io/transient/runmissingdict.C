{
   gSystem->Load("libCintex"); ROOT::Cintex::Cintex::Enable();
   gROOT->ProcessLine(".L missingdict_rflx.cpp+");
   TClass::GetClass("Content")->GetStreamerInfo();
   TClass::GetClass("TransientHolder")->GetStreamerInfo();
   return 0;
}
