{
// Fill out the code of the actual test
gROOT->ProcessLine(".L NtpLib.C+");
FillNtp();
TFile *_file0 = TFile::Open("NtpRecord.root"); 
TTree *tree = (TTree*)_file0->Get("Ntp");

tree->Scan("fEvents.fNShower");
tree->Scan("fEvents[].fShwInd");
tree->Scan("fEvents.fEventNo");
tree->Scan("fEvents.fEventNo:fEvents.fShwInd");
tree->Scan("fEvents.fEventNo*100+fEvents[].fShwInd[]");
tree->Scan("fEvents.fEventNo:fEvents[].fShwInd[]","fEvents[].fNShower > 0");
tree->Scan("fEvents.fEventNo:fEvents[].fShwInd[]:fShowers","fEvents[].fNShower > 0");
tree->Scan("fEvents.fEventNo:fEvents[].fShwInd[]:fShowers[fEvents[].fShwInd[0]].fEnergy","fEvents[].fNShower > 0");
tree->Scan("fShowers[fEvents[].fShwInd[0]].fEnergy","fEvents[].fNShower > 0");
tree->Scan("fEvents.fEventNo:fShowers[fEvents[].fShwInd[]].fEnergy","");

return;
// Wrong
tree->Scan("fShowers[fEvents[].fShwInd[0]].fEnergy");
tree->Scan("fEvents[].fShwInd[0]");
tree->Scan("fShowers[fEvents[].fShwInd[]].fEnergy","fEvents[].fNShower > 0");
tree->Scan("fEvents[].fNShower:fShowers[fEvents[].fShwInd[0]].fEnergy"); 
tree->Scan("fEvents[].fNShower:fShowers[fEvents[].fShwInd[]].fEnergy","1");


}
