#if defined(ClingWorkAroundBrokenUnnamedReturn)
int rundeepClass()
#endif
{
  TFile OriginalFile("ver_40200.root");
  TFile CopyFile("CopyTree.root");

  TTree* OriginalTree = (TTree*) OriginalFile.Get("NtpSt");
  TTree* CopyTree     = (TTree*) CopyFile.Get("NtpSt"); 

  OriginalTree->Scan("fHeader.fVldContext.fDetector:fHeader.fVldContext.fSimFlag:fHeader.fVldContext.fTimeStamp.fSec","","colsize=20",10);
  CopyTree->Scan("fHeader.fVldContext.fDetector:fHeader.fVldContext.fSimFlag:fHeader.fVldContext.fTimeStamp.fSec","","colsize=20",10);

  return 0;  
}

