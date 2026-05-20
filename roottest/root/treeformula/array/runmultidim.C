{
// Fill out the code of the actual test
gROOT->ProcessLine(".L NtpLib.C+");
#ifdef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine("FillNtp();");
#else
FillNtp();
#endif
TFile *_file0 = TFile::Open("NtpRecord.root"); 
if (_file0)
{
TTree *tree = (TTree*)_file0->Get("Ntp");

tree->Scan("fEvents.fNShower","","colsize=20");
tree->Scan("fEvents[].fShwInd","","colsize=20");
tree->Scan("fEvents.fEventNo","","colsize=20");
tree->Scan("fEvents.fEventNo:fEvents.fShwInd","","colsize=20");
tree->Scan("fEvents.fEventNo*100+fEvents[].fShwInd[]","","col=03.9");
tree->Scan("fEvents.fEventNo:fEvents[].fShwInd[]","fEvents[].fNShower > 0","colsize=20");
tree->Scan("fEvents.fEventNo:fEvents[].fShwInd[]:fShowers","fEvents[].fNShower > 0","colsize=20");
tree->Scan("fEvents[].fShwInd[0]","fEvents[].fNShower > 0","colsize=20");
tree->Scan("fEvents.fEventNo:fShowers[fEvents[].fShwInd[]].fEnergy","","colsize=20");
tree->Scan("fEvents.fEventNo*1000+fEvents[].fShwInd[]*100:fShowers[fEvents[].fShwInd[0]].fEnergy","","colsize=20");
tree->Scan("fEvents[].fNShower:fShowers[fEvents[].fShwInd[0]].fEnergy","","colsize=20"); 
tree->Scan("fShowers[fEvents[].fShwInd[0]].fEnergy","","colsize=20");
tree->Scan("fEvents[].fShwInd[0]","","colsize=20");


tree->Scan("fEvents.fEventNo:fEvents[].fNShower:fShowers[fEvents[].fShwInd[]].fEnergy","fEvents[].fShwInd[]*0==0","colsize=20");
}
return;
/*
// Wrong
tree->Scan("fShowers[fEvents[].fShwInd[]].fEnergy","fEvents[].fNShower > 0"); // to fix remove condition
tree->Scan("fEvents[].fNShower:fShowers[fEvents[].fShwInd[]].fEnergy","1"); // to fix replace cond : fEvents[].fShwInd[]*0==0

tree->Scan("fEvents.fEventNo:fEvents[].fShwInd[]:fShowers[fEvents[].fShwInd[0]].fEnergy","fEvents[].fNShower > 0"); // can NOT find a correction!
tree->Scan("fShowers[fEvents[].fShwInd[0]].fEnergy","fEvents[].fNShower > 0","colsize=20"); // to fix remove condition

// wrong size:
root [55]   tree->Scan("fEvents.fEventNo:fShowers[fEvents[].fShwInd[]].fEnergy","fEvents[].fShwInd[]*0==0","col=18");    
*******************************************************
*    Row   * Instance *   fEvents.fEventNo * fShowers *
*******************************************************
*        0 *        0 *                  0 *       10 *
*        0 *        1 *                  0 *       20 *
*        0 *        2 *                  2 *       30 *
*******************************************************
*/

}
