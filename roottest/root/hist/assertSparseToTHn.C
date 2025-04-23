int assertSparseToTHn()
{
  TFile *_file0 = TFile::Open("./effectiveCorrection.root");
  THnSparseF* sp = static_cast<THnSparseF*>(_file0->Get("pbar_Corr0"));
  sp->Dump();
  THn *hh = THn::CreateHn("foo","bar",sp);
  hh->Dump();
  return sp->GetEntries() == hh->GetEntries() ? 0 : 1;
}  
