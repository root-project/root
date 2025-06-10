{
   gROOT->ProcessLine(".L little.C+");
   TClass::GetClass("little")->GetStreamerInfo()->ls("noaddr");
}
