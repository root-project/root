void Build_Timing_Tree(TFile * f, const Char_t *pattern, Int_t& max_slaves);

void Draw_PerfProfiles(const Char_t* filename) {
   // Plots total processing time as a function of number of slaves
   // using each of the 3 selectors.

   // filename: name of file that Run_Node_Tests.C wrote its output into

   gROOT->SetStyle("Plain");
   gStyle->SetOptStat(0);
   gStyle->SetNdivisions(505);
   gStyle->SetTitleFontSize(0.07);

   if(!TString(gSystem->GetLibraries()).Contains("Proof"))
      gSystem->Load("libProof.so");

   TFile f(filename);
   if (f.IsZombie()) {
      cout << "file " << filename << " cannot be opened" << endl;
      return;
   }

   TString perfstats_name = "PROOF_PerfStats";

   Int_t ns_holder;
   Int_t run_holder;
   Float_t time_holder;

   Int_t procmax_slaves = 0;
   TTree* tt_proc    = Build_Timing_Tree(&f, perfstats_name+"_Proc_",
                                         procmax_slaves);
   tt_proc->SetMarkerStyle(4);
   //set branch addresses
   tt_proc->GetBranch("perfproctime")->GetLeaf("nslaves")->SetAddress(&ns_holder);
   tt_proc->GetBranch("perfproctime")->GetLeaf("run")->SetAddress(&run_holder);
   tt_proc->GetBranch("perfproctime")->GetLeaf("time")->SetAddress(&time_holder);

   Int_t procoptmax_slaves = 0;
   TTree* tt_procopt = Build_Timing_Tree(&f, perfstats_name+"_ProcOpt_",
                                         procoptmax_slaves);
   tt_procopt->SetMarkerStyle(5);
   //set branch addresses
   tt_procopt->GetBranch("perfproctime")->GetLeaf("nslaves")->SetAddress(&ns_holder);
   tt_procopt->GetBranch("perfproctime")->GetLeaf("run")->SetAddress(&run_holder);
   tt_procopt->GetBranch("perfproctime")->GetLeaf("time")->SetAddress(&time_holder);

   Int_t noprocmax_slaves = 0;
   TTree* tt_noproc  = Build_Timing_Tree(&f, perfstats_name+"_NoProc_",
                                         noprocmax_slaves);
   tt_noproc->SetMarkerStyle(6);
   //set branch addresses
   tt_noproc->GetBranch("perfproctime")->GetLeaf("nslaves")->SetAddress(&ns_holder);
   tt_noproc->GetBranch("perfproctime")->GetLeaf("run")->SetAddress(&run_holder);
   tt_noproc->GetBranch("perfproctime")->GetLeaf("time")->SetAddress(&time_holder);

   f.Close();

   Int_t nslaves = procmax_slaves>procoptmax_slaves?procmax_slaves:procoptmax_slaves;
   if (nslaves<noprocmax_slaves) nslaves=noprocmax_slaves;

   TProfile* procprof = new TProfile("procprof", "Total Processing Time",
                                     nslaves+1, 0, nslaves+1);
   procprof->SetMarkerStyle(26);
   tt_proc->Draw("time:nslaves>>procprof");
   procprof->GetXaxis()->SetTitle("Number of Slaves");
   procprof->GetYaxis()->SetTitle("Processing Time [s]");

   TProfile* procoptprof = new TProfile("procoptprof", "Total Processing Time",
                                        nslaves+1, 0, nslaves+1);
   procoptprof->SetMarkerStyle(25);
   tt_procopt->Draw("time:nslaves>>procoptprof","","same");

   TProfile* noprocprof = new TProfile("noprocprof", "Total Processing Time",
                                       nslaves+1, 0, nslaves+1);
   noprocprof->SetMarkerStyle(24);
   tt_noproc->Draw("time:nslaves>>noprocprof","","same");

   Float_t lm = gPad->GetLeftMargin();
   Float_t rm = gPad->GetRightMargin();
   Float_t tm = gPad->GetTopMargin();
   Float_t bm = gPad->GetBottomMargin();

   Float_t legxoffset = 0.1;
   Float_t legwidth = 0.2;
   Float_t legyoffset = 0.02;
   Float_t legheight = 0.15;

   TLegend* leg = new TLegend(lm+legxoffset*(1.0-lm-rm),
                              1.0-tm-(legyoffset+legheight)*(1.0-tm-bm),
                              lm+(legxoffset+legwidth)*(1.0-lm-rm),
                              1.0-tm-legyoffset*(1.0-tm-bm));
   leg->SetBorderSize(1);
   leg->SetFillColor(0);
   leg->AddEntry(procprof,"Full Event","p");
   leg->AddEntry(procoptprof,"Partial Event","p");
   leg->AddEntry(noprocprof,"No Data","p");
   leg->Draw();

   gPad->Update();
   TPaveText* titlepave = dynamic_cast<TPaveText*>(gPad->GetListOfPrimitives()->FindObject("title"));
   if (titlepave) {
      Double_t x1ndc = titlepave->GetX1NDC();
      Double_t x2ndc = titlepave->GetX2NDC();
      titlepave->SetX1NDC((1.0-x2ndc+x1ndc)/2.);
      titlepave->SetX2NDC((1.0+x2ndc-x1ndc)/2.);
      titlepave->SetBorderSize(0);
      gPad->Update();
   }
   gPad->Modified();
}

TTree* Build_Timing_Tree(TFile * f, const Char_t *pattern, Int_t& max_slaves) {

   TTree* timing_tree = new TTree("Timing Tree", "Timing Tree");
   timing_tree->SetDirectory(0);
   Int_t ns_holder;
   Int_t run_holder;
   Float_t time_holder;
   TBranch* br = timing_tree->Branch("perfproctime", &ns_holder,
                                     "nslaves/I:run/I:time/F");
   br->GetLeaf("nslaves")->SetAddress(&ns_holder);
   br->GetLeaf("run")->SetAddress(&run_holder);
   br->GetLeaf("time")->SetAddress(&time_holder);

   // extract timing info
   max_slaves = 0;
   TIter NextKey(f->GetListOfKeys());
   TKey* key = 0;
   while (key = dynamic_cast<TKey*>(NextKey())) {
      if(!TString(key->GetName()).Contains(TRegexp(pattern)))
         continue;

      TObject* obj = key->ReadObj();
      TTree* t = dynamic_cast<TTree*>(obj);
      if (!t) {
         delete obj;
         continue;
      }

      //parse name to get number of slaves and run
      Int_t Index = 0;
      const Char_t *name = t->GetName();
      while (Index<strlen(name)) {
        if ( name[Index]>='0' && name[Index]<='9')
        break;
        Index++;
      }

      if (Index == strlen(name)) {
         delete t;
         continue;
      } else {
         // this should be the number of slaves
         ns_holder = atoi(name+Index);
      }

      // get past number of slaves
      while (Index<strlen(name)) {
        if ( name[Index]<'0' || name[Index]>'9')
        break;
        Index++;
      }

      if (Index == strlen(name)) {
         delete t;
         continue;
      }

      while (Index<strlen(name)) {
        if ( name[Index]>='0' && name[Index]<='9')
        break;
        Index++;
      }

      if (Index == strlen(name)) {
         delete t;
         continue;
      } else {
         // this should be the run number
         run_holder = atoi(name+Index);
      }

      if(!t->FindBranch("PerfEvents")) {
         delete t;
         continue;
      }

      // extract timing information
      TPerfEvent pe;
      TPerfEvent* pep = &pe;
      t->SetBranchAddress("PerfEvents",&pep);
      Long64_t entries = t->GetEntries();
      Double_t start, end;
      Bool_t started=kFALSE;
      for (Long64_t k=0; k<entries; k++) {
         t->GetEntry(k);
         if (!started) {
            if (pe.fType==TVirtualPerfStats::kPacket) {
               start = pe.fTimeStamp.GetSec()
                       + 1e-9*pe.fTimeStamp.GetNanoSec()
                       - pe.fProcTime;
               started=kTRUE;
            }
         } else {
            if (pe.fType==TVirtualPerfStats::kPacket) {
               end = pe.fTimeStamp.GetSec()
                     + 1e-9*pe.fTimeStamp.GetNanoSec();
            }
         }
      }

      time_holder = end-start;
      timing_tree->Fill();
      if (max_slaves<ns_holder) max_slaves=ns_holder;

      delete t;
   }

   return timing_tree;
}
