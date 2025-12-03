{
   gROOT->ProcessLine(".L classes.C+");
   gROOT->ProcessLine(".L minostest.C+");
   TClass::GetClass("SEIdAltL")->GetStreamerInfo()->ls("noaddr");
}
