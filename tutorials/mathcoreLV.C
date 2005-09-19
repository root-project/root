
/// Cint macro to test writing of mathcore Lorentz Vectors in a Tree

void mathcoreLV() { 

  
  int nEvents = 10000;

  write(nEvents);

  read();
}


void write(int n) { 

  gSystem->Load("libMathCore");  
  gSystem->Load("libPhysics");  
  using namespace ROOT::Math;

  TRandom R; 
  TStopwatch timer;


  TFile f1("mathcoreLV.root","RECREATE");

  // create tree
  TTree t1("t1","Tree with new LorentzVector");

  std::vector<ROOT::Math::XYZTVector>  tracks; 
  std::vector<ROOT::Math::XYZTVector> * pTracks = &tracks; 
  t1.Branch("tracks","std::vector<XYZTVector>",&pTracks);

  timer.Start();
  for (int i = 0; i < n; ++i) { 
    int nPart = R.Poisson(5);
    pTracks->clear();
    pTracks->reserve(nPart); 
    for (int j = 0; j < nPart; ++j) {
      double Px = R.Gaus(0,10);
      double Py = R.Gaus(0,10);
      double Pz = R.Gaus(0,10);
      double E  = TMath::Max(R.Gaus(100,30),0.0);
      XYZTVector v1(Px,Py,Pz,E);
      pTracks->push_back(v1);
    }
    t1.Fill(); 
  }

  f1.Write();
  timer.Stop();
  std::cout << " Time for new Vector " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 

  t1.Print();
}


void read() { 

  gSystem->Load("libMathCore");  
  gSystem->Load("libPhysics");  
  using namespace ROOT::Math;

  TRandom R; 
  TStopwatch timer;

  TH1D * h1 = new TH1D("h1","total energy  of event ",100,0,1000);
  TH1D * h2 = new TH1D("h2","Track Energy",100,0,200);
  TH1D * h3 = new TH1D("h2","Number of track per event",20,0,20);


  TFile f1("mathcoreLV.root");

  // create tree
  TTree *t1 = (TTree*)f1.Get("t1");

  std::vector<ROOT::Math::XYZTVector> * pTracks = 0;
  t1->SetBranchAddress("tracks",&pTracks);

  timer.Start();
  int n = (int) t1->GetEntries();
  std::cout << " Tree Entries " << n << std::endl; 
  double etot=0;
  for (int i = 0; i < n; ++i) { 
    t1->GetEntry(i);
    int ntrk = pTracks->size(); 
    h3->Fill(ntrk);
    XYZTVector q; 
    for (int j = 0; j < ntrk; ++j) { 
      XYZTVector v = (*pTracks)[j]; 
      q += v; 
      h2->Fill(v.E());
    }
    h1->Fill(q.M() );
  }


  timer.Stop();
  std::cout << " Time for new Vector " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 


  
  TCanvas *c1 = new TCanvas("c1","demo of Trees",10,10,600,800);
  c1->Divide(1,3);
  c1->cd(1);
  h1->Draw();
  c1->cd(2);
  h2->Draw();
  c1->cd(3);
  h3->Draw();

}


  
