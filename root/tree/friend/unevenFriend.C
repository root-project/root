{

TChain *chain = new TChain ("Global");
chain->Add("short0.root");
chain->Add("short1.root");
TChain *friend = new TChain ("Global");
friend->Add("long.root");                             

//chain->Draw("TrigL1CJT[0]","",""); // that works
//chain->Draw("TrigL1CJT[0]","TrigL1CJT[0]==1",""); //doen't work

chain->LoadTree(0);
chain->AddFriend(friend,"friendglobal");
chain->Draw("TrigL1CJT[0]","TrigL1CJT[0]==1",""); //doen't work

if (htemp->GetMean()!=1) {
   fprintf(stdout,"histogram must has wrong mean\n");
   gApplication->Terminate(1);
} else {
   fprintf(stdout,"histogram must have been correctly created\n");
}

}
