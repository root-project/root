void mlpHiggs(Int_t ntrain=100) {
// Example of a Multi Layer Perceptron
// For a LEP search for invisible Higgs boson, a neural network 
// was used to separate the signal from the background passing 
// some selection cuts. Here is a simplified version of this network, 
// taking into account only WW events.
//Author: Christophe Delaere
   
   if (!gROOT->GetClass("TMultiLayerPerceptron")) {
      gSystem->Load("libMLP");
   }

   // Prepare inputs
   // The 2 trees are merged into one, and a "type" branch, 
   // equal to 1 for the signal and 0 for the background is added.
   const char *fname = "mlpHiggs.root";
   TFile *input = 0;
   if (!gSystem->AccessPathName(fname)) {
      input = TFile::Open(fname);
   } else {
      printf("accessing %s file from http://root.cern.ch/files\n",fname);
      input = TFile::Open(Form("http://root.cern.ch/files/%s",fname));
   }
   if (!input) return;

   TTree *signal = (TTree *) input->Get("sig_filtered");
   TTree *background = (TTree *) input->Get("bg_filtered");
   TTree *simu = new TTree("MonteCarlo", "Filtered Monte Carlo Events");
   Float_t ptsumf, qelep, nch, msumf, minvis, acopl, acolin;
   Int_t type;
   signal->SetBranchAddress("ptsumf", &ptsumf);
   signal->SetBranchAddress("qelep",  &qelep);
   signal->SetBranchAddress("nch",    &nch);
   signal->SetBranchAddress("msumf",  &msumf);
   signal->SetBranchAddress("minvis", &minvis);
   signal->SetBranchAddress("acopl",  &acopl);
   signal->SetBranchAddress("acolin", &acolin);
   background->SetBranchAddress("ptsumf", &ptsumf);
   background->SetBranchAddress("qelep",  &qelep);
   background->SetBranchAddress("nch",    &nch);
   background->SetBranchAddress("msumf",  &msumf);
   background->SetBranchAddress("minvis", &minvis);
   background->SetBranchAddress("acopl",  &acopl);
   background->SetBranchAddress("acolin", &acolin);
   simu->Branch("ptsumf", &ptsumf, "ptsumf/F");
   simu->Branch("qelep",  &qelep,  "qelep/F");
   simu->Branch("nch",    &nch,    "nch/F");
   simu->Branch("msumf",  &msumf,  "msumf/F");
   simu->Branch("minvis", &minvis, "minvis/F");
   simu->Branch("acopl",  &acopl,  "acopl/F");
   simu->Branch("acolin", &acolin, "acolin/F");
   simu->Branch("type",   &type,   "type/I");
   type = 1;
   Int_t i;
   for (i = 0; i < signal->GetEntries(); i++) {
      signal->GetEntry(i);
      simu->Fill();
   }
   type = 0;
   for (i = 0; i < background->GetEntries(); i++) {
      background->GetEntry(i);
      simu->Fill();
   }
   // Build and train the NN ptsumf is used as a weight since we are primarly 
   // interested  by high pt events.
   // The datasets used here are the same as the default ones.
   TMultiLayerPerceptron *mlp = 
      new TMultiLayerPerceptron("@msumf,@ptsumf,@acolin:5:3:type",
                                "ptsumf",simu,"Entry$%2","(Entry$+1)%2");
   mlp->Train(ntrain, "text,graph,update=10");
   mlp->Export("test","python");
   // Use TMLPAnalyzer to see what it looks for
   TCanvas* mlpa_canvas = new TCanvas("mlpa_canvas","Network analysis");
   mlpa_canvas->Divide(2,2);
   TMLPAnalyzer ana(mlp);
   // Initialisation
   ana.GatherInformations();
   // output to the console
   ana.CheckNetwork();
   mlpa_canvas->cd(1);
   // shows how each variable influences the network
   ana.DrawDInputs();
   mlpa_canvas->cd(2);
   // shows the network structure
   mlp->Draw();
   mlpa_canvas->cd(3);
   // draws the resulting network
   ana.DrawNetwork(0,"type==1","type==0");
   mlpa_canvas->cd(4);
   // Use the NN to plot the results for each sample
   // This will give approx. the same result as DrawNetwork.
   // All entries are used, while DrawNetwork focuses on 
   // the test sample. Also the xaxis range is manually set.
   TH1F *bg = new TH1F("bgh", "NN output", 50, -.5, 1.5);
   TH1F *sig = new TH1F("sigh", "NN output", 50, -.5, 1.5);
   bg->SetDirectory(0);
   sig->SetDirectory(0);
   Double_t params[4];
   for (i = 0; i < background->GetEntries(); i++) {
      background->GetEntry(i);
      params[0] = msumf;
      params[1] = ptsumf;
      params[2] = acolin;
      params[3] = acopl;
      bg->Fill(mlp->Evaluate(0, params));
   }
   for (i = 0; i < signal->GetEntries(); i++) {
      signal->GetEntry(i);
      params[0] = msumf;
      params[1] = ptsumf;
      params[2] = acolin;
      params[3] = acopl;
      sig->Fill(mlp->Evaluate(0,params));
   }
   bg->SetLineColor(kBlue);
   bg->SetFillStyle(3008);   bg->SetFillColor(kBlue);
   sig->SetLineColor(kRed);
   sig->SetFillStyle(3003); sig->SetFillColor(kRed);
   bg->SetStats(0);
   sig->SetStats(0);
   bg->Draw();
   sig->Draw("same");
   TLegend *legend = new TLegend(.75, .80, .95, .95);
   legend->AddEntry(bg, "Background (WW)");
   legend->AddEntry(sig, "Signal (Higgs)");
   legend->Draw();
   mlpa_canvas->cd(0);
   delete input;
}
