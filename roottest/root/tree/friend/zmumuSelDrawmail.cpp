#include <TROOT.h>
#include <TTree.h>
#include <TBranch.h>
#include <TChain.h>
#include <TFile.h>

#include <Riostream.h>

#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TPostScript.h"
#include "TCut.h"


TTree* zfriendtree;

void zmumuSelDraw(TTree* t0){

   zfriendtree=new TTree("friendextra","A friend tree for new branches");

   zfriendtree->SetDirectory(0);
   /*
     void SetDirectory(TDirectory *dir) 
     Remove reference to this tree from current directory and add
     reference to new directory dir. dir can be 0 in which case the tree
     does not belong to any directory.
     
     There is a bug in the Friend implementation. If the tree (to which you
     add friend to) is memory resident it's directory has to be zero. (See
     work-around below).  TTree* zfriendtree=new TTree("friendextra","A friend
     tree for new branches"); zfriendtree is meant to be in memory (right?)
     zfriendtree->SetDirectory(0);
   */


   Double_t neumomcmtot; 
   zfriendtree->Branch("neumomcmtot",&neumomcmtot,"neumomcmtot/D");

   TTree *zChain=t0;
   Int_t fCurrent=-1;
   zChain->SetMakeClass(1);
   Int_t nneu;
   Float_t neumomcm[40];//={0.0,};

   zChain->SetBranchAddress("nneu",&nneu);
   zChain->SetBranchAddress("neumomcm",neumomcm);


   TBranch *b_nneu=zChain->GetBranch("nneu");
   TBranch *b_neumomcm=zChain->GetBranch("neumomcm");
   
   Long64_t nentries = zChain->GetEntriesFast();
   
   for (Long64_t jentry=0; jentry<nentries;jentry++) {

      if(!zChain) break; 

      Long64_t centry = zChain->LoadTree(jentry);

      if(centry<0 || (zChain->IsA() != TChain::Class())) break;

      TChain *chain =(TChain*)zChain;

      if(chain->GetTreeNumber() !=fCurrent){
         fCurrent = chain->GetTreeNumber();
         b_nneu = zChain->GetBranch("nneu");
         b_neumomcm = zChain->GetBranch("neumomcm");
      }       
      
      double check = 0.0;
      neumomcmtot = 0.0;

      b_nneu->GetEntry(centry);
      b_neumomcm->GetEntry(centry);

      if(nneu>40){
         cout<<"Number of neutral track is over 40"<<endl;
         break;
      }
      for(int j=0;j<nneu;j++){
         check +=  neumomcm[j];
      }
      neumomcmtot = check;
      cout<< jentry << ": neumomcmtot: "<<neumomcmtot<<endl;
      zfriendtree->Fill();
   }


   
   zfriendtree->AddFriend(t0);
   zfriendtree->SetScanField(-1);
   //return;


   // this used to crash;
   zfriendtree->Scan("nneu:neumomcm[0]:neumomcm[1]:neumomcm[2]:neumomcm[3]:neumomcmtot");
   zfriendtree->Scan("nneu:neumomcm:neumomcmtot","","",8,0);
   zfriendtree->Scan("nneu:npi0:neumomcm:pi0thecm","","",8,0);
   zfriendtree->Scan("nneu:npi0:neumomcm:pi0thecm","neumomcm[0]>0","",8,0);
   zfriendtree->Scan("nneu:npi0:neumomcm:pi0thecm","neumomcm>0","",8,0);

   zfriendtree->Scan("nneu:neumomcmtot-(Alt$(neumomcm[0],0)+Alt$(neumomcm[1],0)+Alt$(neumomcm[2],0)+Alt$(neumomcm[3],0))");
   zfriendtree->Scan("nneu:neumomcmtot-(Alt$(neumomcm[0],0)+Alt$(neumomcm[1],0)+Alt$(neumomcm[2],0)+Alt$(neumomcm[3],0))","nneu==0");
   zfriendtree->Scan("nneu:neumomcmtot-(Alt$(neumomcm[0],0)+Alt$(neumomcm[1],0)+Alt$(neumomcm[2],0)+Alt$(neumomcm[3],0))","nneu==1");

   zfriendtree->Scan("nneu:neumomcm[0]:neumomcm[1]:neumomcm[2]:neumomcm[3]:neumomcmtot:neumomcmtot-(Alt$(neumomcm[0],0)+Alt$(neumomcm[1],0)+Alt$(neumomcm[2],0)+Alt$(neumomcm[3],0))","nneu>1&&nneu<=4");
   zfriendtree->Scan("nneu:neumomcmtot-(Alt$(neumomcm[0],0)+Alt$(neumomcm[1],0)+Alt$(neumomcm[2],0)+Alt$(neumomcm[3],0))","nneu>4");
   
   zfriendtree->RemoveFriend(t0);
    

   // Test a chain with a TTree as friend
   t0->AddFriend(zfriendtree);
//    cout << "(TTree*)" << (void*)zfriendtree << endl;

//    cout << "t0 has " << t0->GetEntries() << " entries\n";
//    cout << "zfriendtree has " << zfriendtree->GetEntries() << " entries\n";
//    t0->ls();

   t0->LoadTree(100);
   if (zfriendtree->GetReadEntry()!=100) {
      cerr << "friend tree not loaded properly " << zfriendtree->GetReadEntry() << " instead of 100 " << endl;
   }

   t0->LoadTree(103);
   if (zfriendtree->GetReadEntry()!=103) {
      cerr << "friend tree not loaded properly " << zfriendtree->GetReadEntry() << " instead of 103 " << endl;
   }

//    zfriendtree->Scan("neumomcmtot");
//    t0->Scan("nneu:neumomcm[0]:neumomcm[1]:neumomcm[2]:neumomcm[3]:neumomcmtot");
}

void zmumuSelDrawmail(){
   TChain* zch=new TChain("TauMicroFilter/ntp1");
   zch->Add("data12_Feb2000_h1900_off-3-100.root");
   zch->Add("data12_Apr2001_h1930_on-84-100.root");
   // zch->Add("musp5_02_02-16.root");
   zmumuSelDraw(zch);
   return; // zch;
   
}
