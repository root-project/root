{
   TFile::Open("lariat-si.root");
   TClass::GetClass("artdaq::QuickVec<unsigned long long>")->GetStreamerInfo()->ls();
   TClass::GetClass("artdaq::Fragment")->GetStreamerInfo()->ls();
}
