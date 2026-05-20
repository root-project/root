{

TChain *chain = new TChain ("Global");
chain->Add("short0.root");
chain->Add("short1.root");
TChain *friendChain = new TChain ("Global");
friendChain->Add("long.root");                             

//chain->Draw("TrigL1CJT[0]","",""); // that works
//chain->Draw("TrigL1CJT[0]","TrigL1CJT[0]==1",""); //doen't work

chain->LoadTree(0);
chain->AddFriend(friendChain,"friendglobal");
chain->Draw("TrigL1CJT[0]","TrigL1CJT[0]==1",""); //doen't work
#ifdef ClingWorkAroundMissingDynamicScope
   TH1F *htemp; htemp = (TH1F*)gROOT->FindObject("htemp");
#endif
if (htemp->GetMean()!=1) {
   fprintf(stdout,"histogram must has wrong mean\n");
   gApplication->Terminate(1);
} else {
   fprintf(stdout,"histogram must have been correctly created\n");
}
#ifndef ClingWorkAroundBrokenUnnamedReturn
return 0;
#endif
}
