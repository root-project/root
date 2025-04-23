{
   gSystem->Load("libRIO");
#ifndef __CLING__
   gROOT->ProcessLine(".exception");
#else
   printf("G__catchexception=0\n");
#endif
#ifdef ClingWorkAroundMissingDynamicScope
   if (TFile *f = (TFile*)gROOT->ProcessLine("gFile;"))
#else
      if (gFile!=0)
#endif         
   {
#ifdef ClingWorkAroundMissingDynamicScope
      TTree *mk; f->GetObject("mk",mk);
#endif
      mk->Scan("iem.mt:iem.i:Emcl[iem.i]._HMx8:iem.dphi_met","(iem.dphi_met>0.5) && (Emcl[iem.i]._HMx8<20)","colsize=18",2,211);
      mk->Scan("iem.mt:iem.i:Emcl[iem.i]._HMx8:iem.dphi_met","(Emcl[iem.i]._HMx8<20) && (iem.dphi_met>0.5)","colsize=18",2,211);
      return;
   }
   TFile *_file0 = TFile::Open("mksm.root");
#ifdef ClingWorkAroundMissingDynamicScope
   TTree *mk; _file0->GetObject("mk",mk);
#endif
   mk->AddFriend("TMBTree","sm.root");
   Long64_t res01 = mk->Draw("iem.mt>>h_iem_Emcl","(iem.dphi_met>0.5) && (Emcl[iem.i]._HMx8<20)");
   //int res01 = mk->Draw("iem.mt>>h_iem_Emcl","(iem.dphi_met>0.5) && (Emcl[iem.i]._HMx8<20)");
   Long64_t res02 = mk->Draw("iem.mt>>h_Emcl_iem","(Emcl[iem.i]._HMx8<20) && (iem.dphi_met>0.5)");
   if (res01!=res02) {
      cout << "Error: order of selections affects the results " << res01 << " vs " << res02 << "\n";
   }
///*
//mk->Scan("iem.mt:iem.i:Emcl[iem.i]._HMx8:iem.dphi_met","(iem.dphi_met>0.5) && (Emcl[iem.i]._HMx8<20)","colsize=15",50,200)
//mk->Scan("iem.mt:iem.i:Emcl[iem.i]._HMx8:iem.dphi_met","(Emcl[iem.i]._HMx8<20) && (iem.dphi_met>0.5)","colsize=15",50,200)
//*/
}
