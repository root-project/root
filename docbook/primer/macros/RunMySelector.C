{// set up a TChain
TChain *ch=new TChain("cond_data", "My Chain for Example N-Tuple");
 ch->Add("conductivity_experiment*.root");
// eventually, start Proof Lite on cores
TProof::Open("workers=4");
ch->SetProof();
ch->Process("MySelector.C+");}
