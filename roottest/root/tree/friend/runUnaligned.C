#include "TTree.h"
#include "TCanvas.h"
#include "Riostream.h"

TTree *createTree() {
  int run = 0;
  const unsigned NCHANS = 2000;
  int channel[NCHANS];
  for(int i = 0; i < NCHANS; i++)
    channel[i] = i;
  
  TTree* tree1 = new TTree("tree1","tree1");
  tree1->Branch("run",&run,"run/I");
  tree1->Branch("channel", channel, "channel[2000]/I");
  
  TTree* tree2 = new TTree("tree2","tree2");
  tree2->Branch("run", &run,"run/I");
  tree2->Branch("channel", channel, "channel[2000]/I");
  
  //fill the tree so the run numbers don't match
  const int nruns = 200;
  for(int i=0; i<(nruns - 20); i++){
    run = i;          // Range from 0 to 179 (inclusive)
    tree1->Fill();
    run = nruns - i;  // Range from 21 to 200 (inclusive)
    tree2->Fill();
  }

  //build the indexes
  tree1->BuildIndex("run");
  tree2->BuildIndex("run");
  
  tree1->AddFriend(tree2, "tree2");
  tree2->ResetBranchAddresses();
  tree1->ResetBranchAddresses();
  return tree1;
}

void runUnaligned()
{
   TTree *tree1 = createTree();

   //now do some drawing
   TCanvas* c1 = new TCanvas;
   c1->Divide(2,2);
   c1->cd(1);
   int n1 = tree1->Draw("channel : tree2.channel","1");
   
   c1->cd(2);
   int n2 = tree1->Draw("channel : tree2.channel","channel == tree2.channel");
   //automatically true
   
   c1->cd(3);
   int n3 = tree1->Draw("channel : tree2.channel", "run == tree2.run");
   // We use to have a problem with the first time we read the entries,
   // so we test that reading twice gives twice the same result!
   int n4 =  tree1->Draw("channel : tree2.channel", "run == tree2.run");
   
   c1->cd(4);
   int n5 = tree1->Draw("channel : tree2.channel", 
                        "run == tree2.run && channel == tree2.channel");
   
   cout<<n1<<' '<<n2<<' '<<n3<<' '<<n4<<' '<<n5<<'\n';
}

/*
 root [21] t = new TTreeFormula("tt","tree2.channel == channel && tree2.run",tree1);
 root [22] t->EvalInstance(2)
 (Double_t)1.00000000000000000e+00
 root [23] t = new TTreeFormula("tt","tree2.channel == channel && tree2.run",tree1); t->GetNdata()
 root [24] 
 root [24] t->EvalInstance(2)
 (Double_t)0.00000000000000000e+00
 
 root [25] t = new TTreeFormula("tt","tree2.channel == channel && run",tree1);
 root [26] t->EvalInstance(2)
 (Double_t)0.00000000000000000e+00
 root [27] t = new TTreeFormula("tt","tree2.channel == channel && run",tree1); t->GetNdata()
 (Int_t)2000
 root [28] t->EvalInstance(2)
 (Double_t)0.00000000000000000e+00
 
 root [29] t = new TTreeFormula("tt","tree2.channel == channel",tree1);
 root [30] t->EvalInstance(2)
 (Double_t)1.00000000000000000e+00
 root [31] t = new TTreeFormula("tt","tree2.channel == channel",tree1); t->GetNdata()
 (Int_t)2000
 root [32] t->EvalInstance(2)
 (Double_t)1.00000000000000000e+00
 root [33] 

 
 root [31] t = new TTreeFormula("tt","tree2.channel == channel",tree1); t->GetNdata()
 (Int_t)2000
 root [33] t->GetMultiplicity()
 (const Int_t)2
 
 root [34] t = new TTreeFormula("tt","tree2.channel == channel && run",tree1);
 root [35] t->GetMultiplicity()
 (const Int_t)2
 
 root [36] t = new TTreeFormula("tt","tree2.channel == channel && tree2.run",tree1);
 root [37] t->GetMultiplicity()
 (const Int_t)1

 root [52] tree1->Scan("channel : tree2.channel","channel == tree2.channel");
 ***********************************************
 *    Row   * Instance *  channel  *  tree2.ch *
 ***********************************************
 *        0 *        0 *         0 *         0 *
 *        0 *        1 *         1 *         1 *
 *        0 *        2 *         2 *         2 *

 root [53] tree1->Scan("channel : tree2.channel","channel == tree2.channel && run");
 ***********************************************
 *    Row   * Instance *  channel  *  tree2.ch *
 ***********************************************
 *        1 *        0 *         0 *         0 *
 *        1 *        1 *         1 *         1 *
 *        1 *        2 *         2 *         2 *
 
 root [54] tree1->Scan("channel : tree2.channel","channel == tree2.channel && tree2.run");
 ***********************************************
 *    Row   * Instance *  channel  *  tree2.ch *
 ***********************************************
 *       21 *        0 *         0 *         0 *
 *       21 *        1 *         1 *         1 *
 *       22 *        0 *         0 *         0 *
 

*/
