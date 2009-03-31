void run_selector_tests(TString selector_name, Int_t fps, Int_t niter,
                        TFile* outputfile, TString name_stem,
                        const Char_t *basedir, Int_t stepsize, Int_t start);

void Run_Node_Tests(Int_t fps, Int_t niter, const Char_t *basedir,
                    TFile *outputfile, Int_t stepsize=5, Int_t start=0)
{
   // This script collects performance statistics (to be saved in outputfile)
   // for running a PROOF cluster with an increasing number of slaves. The
   // script runs multiple times for each number of slaves to gather
   // statistics for each number of slaves.

   // fps: number of files to be processed per slave
   // niter: number of iterations to run each test at each number of nodes
   // basedir: base directory where files are located on slave
   //             (same input as given to make_event_trees.C)
   // outputfile: outputfile to write output info to
   // stepsize: number of slaves to increase by when performing tests
   // start: number of slaves of slaves to start test with
   //         (0 means start with stepsize slaves)

   if (!gProof) {
      cout << "PROOF must be started before running this script" << endl;
      return;
   }

   if (fps<1) {
      cout << "Must be testing at least 1 file per slave" << endl;
      return;
   }

   if (niter<1) {
      cout << "Atleast 1 iteration per step is required" << endl;
      return;
   }

   if (stepsize<1) {
      cout << "Must increase subsequent tests by atleast 1 slave" << endl;
      return;
   }

   if (start<0) {
      cout << "starting number of nodes must be atleast 1" << endl
           << "  (with 0 making start = stepsize)" << endl;
      return;
   }

   if (start==0) start = stepsize;

#if 0
   // load Event library
   if (!TString(gSystem->GetLibraries()).Contains("Event")) {
      if(gSystem->Load("$ROOTSYS/test/libEvent.so")) {
         cout << "Could not load libEvent.so" << endl;
         return;
      }
   }

   // add $ROOTSYS/test to include path
   if (!TString(gSystem->GetIncludePath()).Contains("-I$ROOTSYS/test"))
      gSystem->AddIncludePath("-I$ROOTSYS/test");
#endif

   if (gProof->UploadPackage("event")) {
      cout << "Could not upload Event par file to slaves" << endl;
      return;
   }

   if (gProof->EnablePackage("event")) {
      cout << "Could not enable Event library on PROOF slaves" << endl;
      return;
   }

   if (gROOT->LoadMacro("make_tdset.C")) {
      cout << "Could not load make_tdset macro" << endl;
      return;
   }

#if 0
   //use new packetizer
   gProof->AddInput(new TNamed("PROOF_NewPacketizer",""));
#else
   gProof->SetParameter("PROOF_ForceLocal", 1);

#endif

   //run the tests
   run_selector_tests("EventTree_Proc.C++", fps, niter, outputfile, "_Proc", basedir, stepsize, start);
   run_selector_tests("EventTree_ProcOpt.C++", fps, niter, outputfile, "_ProcOpt", basedir, stepsize, start);
   run_selector_tests("EventTree_NoProc.C++", fps, niter, outputfile, "_NoProc", basedir, stepsize, start);

}

void run_selector_tests(TString selector_file, Int_t fps, Int_t niter,
                        TFile* outputfile, TString name_stem,
                        const Char_t *basedir, Int_t stepsize, Int_t start)
{

   TString selector_name(selector_file);
   selector_name.Remove(selector_file.Last('.'));

   gProof->SetParallel(9999);
   Int_t nslaves = gProof->GetParallel();
   if (start > nslaves) {
      cout << "starting number of nodes must be atleast 1" << endl
           << "  (with 0 making start = stepsize)" << endl;
      return;
   }

   //run once on all nodes with no logging
   gEnv->SetValue("Proof.StatsHist",0);
   gEnv->SetValue("Proof.StatsTrace",0);
   gEnv->SetValue("Proof.SlaveStatsTrace",0);
   TDSet* dset = make_tdset(basedir, fps);
   cout << "Running on all nodes first" << endl;
   gProof->Load(selector_file);
   dset->Process(selector_name);
//   delete dset;
   cout << "done" << endl;

   // switch logging on (no slave logging)
   gEnv->SetValue("Proof.StatsHist",0);
   gEnv->SetValue("Proof.StatsTrace",1);
   gEnv->SetValue("Proof.SlaveStatsTrace",0);

   TString perfstats_name = "PROOF_PerfStats";

   // set up timing tree info
   TString time_tree_name = perfstats_name;
   time_tree_name+=name_stem;
   time_tree_name+="_timing_tree";
   TTree timing_tree(time_tree_name,"Timing Tree");
   Int_t ns_holder;
   Int_t run_holder;
   Float_t time_holder;
   TBranch* br = timing_tree.Branch("tests",&ns_holder,"nslaves/I:run/I:time/F");
   if (outputfile && !outputfile->IsZombie())
      timing_tree.SetDirectory(outputfile);
   br->GetLeaf("nslaves")->SetAddress(&ns_holder);
   br->GetLeaf("run")->SetAddress(&run_holder);
   br->GetLeaf("time")->SetAddress(&time_holder);

   Bool_t done = kFALSE;
   for (Int_t nactive=start; !done; nactive+=stepsize) {

      if (nactive >= nslaves) {
         done=kTRUE;
         nactive=nslaves;
      }

      gProof->SetParallel(nactive);
      ns_holder = nactive;
      for (Int_t j=0; j<niter; j++) {
         run_holder=j;
//         dset = make_tdset(basedir,fps);
         TTime starttime = gSystem->Now();
         dset->Process(selector_name);
         TTime endtime = gSystem->Now();
         time_holder = Long_t(endtime-starttime)/Float_t(1000);
         cout << "Processing time was " << time_holder << " seconds" << endl;

         if (outputfile && !outputfile->IsZombie()) {
            TList* l = dset->GetOutputList();

            //save perfstats
            TTree* t = dynamic_cast<TTree*>(l->FindObject(perfstats_name.Data()));
            if (t) {
               TDirectory* trdir = t->GetDirectory();
               TDirectory* dirsav = gDirectory;
               outputfile->cd();
               t->SetDirectory(outputfile);
               TString origname = t->GetName();
               TString newname = perfstats_name;
               newname+=name_stem;
               newname+="_";
               newname+=nactive;
               newname+="slaves_run";
               newname+=j;
               t->SetName(newname);
               t->Write();
               t->SetName(origname);
               t->SetDirectory(trdir);
               dirsav->cd();
            } else {
               cout << perfstats_name.Data() << " tree not found" << endl << flush;
            }
         
            //save outputhistos
            TString ptdist_name = "pt_dist";
            TH1* h = dynamic_cast<TH1*>(l->FindObject(ptdist_name.Data()));
            if (h) {
               TDirectory* hdir = h->GetDirectory();
               TDirectory* dirsav = gDirectory;
               outputfile->cd();
               h->SetDirectory(outputfile);
               TString origname = h->GetName();
               TString newname = ptdist_name;
               newname+=name_stem;
               newname+="_";
               newname+=nactive;
               newname+="slaves_run";
               newname+=j;
               h->SetName(newname);
               h->Write();
               h->SetName(origname);
               h->SetDirectory(hdir);
               dirsav->cd();
            } else {
               cout << ptdist_name.Data() << " histogram not found" << endl << flush;
            }

            TString tracksdist_name = "ntracks_dist";
            TH1* h2 = dynamic_cast<TH1*>(l->FindObject(tracksdist_name.Data()));
            if (h2) {
               TDirectory* hdir = h2->GetDirectory();
               TDirectory* dirsav = gDirectory;
               outputfile->cd();
               h2->SetDirectory(outputfile);
               TString origname = h2->GetName();
               TString newname = tracksdist_name;
               newname+=name_stem;
               newname+="_";
               newname+=nactive;
               newname+="slaves_run";
               newname+=j;
               h2->SetName(newname);
               h2->Write();
               h2->SetName(origname);
               h2->SetDirectory(hdir);
               dirsav->cd();
            } else {
               cout << tracksdist_name.Data() << " histogram not found" << endl << flush;
            }
         }

//         delete dset;
         timing_tree.Fill();
      }
   }

         delete dset;

   if (outputfile && !outputfile->IsZombie()) {
      TDirectory* dirsav=gDirectory;
      outputfile->cd();
      timing_tree.Write();
      timing_tree.SetDirectory(0);
      dirsav->cd();
   }
}

