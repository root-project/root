{

TChain *chain = new TChain ("Global");
chain->Add("d0_analyze1.root");    
chain->Add("d0_analyze.root");        
TChain *friend = new TChain ("Global");
friend->Add("wtaunu.root");                             

//chain->Draw("TrigL1CJT[0]","",""); // that works
//chain->Draw("TrigL1CJT[0]","TrigL1CJT[0]==1",""); //doen't work

chain->LoadTree(0);
chain->AddFriend(friend,"friendglobal");
chain->Draw("TrigL1CJT[0]","TrigL1CJT[0]==1",""); //doen't work

if (htemp->GetMean()!=1) {
   gApplication->Terminate(1);
} else {
   fprintf(stdout,"histogram must have been correctly created\n");
}

}
