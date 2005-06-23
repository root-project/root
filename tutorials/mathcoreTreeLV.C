
/// Cint macro to test writing of mathcore Lorentz Vectors in a Tree

void mathcoreTreeLV() { 

  
  int nEvents = 100000;

  write(nEvents);

  read();
}


void write(int n) { 

  gSystem->Load("libMathCore");  
  gSystem->Load("libPhysics");  
  using namespace ROOT::Math;

  TRandom R; 
  TStopwatch timer;


  timer.Start();
  for (int i = 0; i < n; ++i) { 
        double Px = R.Gaus(0,10);
	double Py = R.Gaus(0,10);
	double Pz = R.Gaus(0,10);
	double E  = R.Gaus(100,10);
  }

  timer.Stop();
  std::cout << " Time for Random gen " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 


  TFile f1("mathcoreTreeLV1.root","RECREATE");

  // create tree
  TTree t1("t1","Tree with new LorentzVector");

  LorentzVector *v1 = new LorentzVector();
  t1.Branch("LV branch","LorentzVector",&v1);

  timer.Start();
  for (int i = 0; i < n; ++i) { 
        double Px = R.Gaus(0,10);
	double Py = R.Gaus(0,10);
	double Pz = R.Gaus(0,10);
	double E  = R.Gaus(100,10);
	//CylindricalEta4D<double> & c = v1->Coordinates();
	//c.SetValues(Px,pY,pZ,E);
	v1->Set(Px,Py,Pz,E);
	t1.Fill(); 
  }

  f1.Write();
  timer.Stop();
  std::cout << " Time for new Vector " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 

  t1.Print();

  // create tree with old LV 

  TFile f2("mathcoreTreeLV2.root","RECREATE");
  TTree t2("t2","Tree with TLorentzVector");

  TLorentzVector * v2 = new TLorentzVector();
  TLorentzVector::Class()->IgnoreTObjectStreamer();
  TVector3::Class()->IgnoreTObjectStreamer();

  t2.Branch("TLV branch","TLorentzVector",&v2,16000,2);

  timer.Start();
  for (int i = 0; i < n; ++i) { 
        double Px = R.Gaus(0,10);
	double Py = R.Gaus(0,10);
	double Pz = R.Gaus(0,10);
	double E  = R.Gaus(100,10);
	v2->SetPxPyPzE(Px,Py,Pz,E);
	t2.Fill(); 
  }

  f2.Write();
  timer.Stop();
  std::cout << " Time for old Vector " << timer.RealTime() << "  " << timer.CpuTime() << endl; 

  t2.Print();
}


void read() { 



  gSystem->Load("libMathCore");  
  gSystem->Load("libPhysics");  
  using namespace ROOT::Math;

  TRandom R; 
  TStopwatch timer;



  TFile f1("mathcoreTreeLV1.root");

  // create tree
  TTree *t1 = (TTree*)f1.Get("t1");

  LorentzVector *v1 = 0;
  t1->SetBranchAddress("LV branch",&v1);

  timer.Start();
  int n = (int) t1->GetEntries();
  std::cout << " Tree Entries " << n << std::endl; 
  double etot=0;
  for (int i = 0; i < n; ++i) { 
    t1->GetEntry(i);
    double Px = v1->Px();
    double Py = v1->Py();
    double Pz = v1->Pz();
    double E = v1->E();
    etot += E;
  }


  timer.Stop();
  std::cout << " Time for new Vector " << timer.RealTime() << "  " << timer.CpuTime() << std::endl; 

  std::cout << " E average" << i<< "  " << etot << "  " << etot/double(i) << endl; 


  // create tree with old LV 

  TFile f2("mathcoreTreeLV2.root");
  TTree *t2 = (TTree*)f2.Get("t2");


  TLorentzVector * v2 = 0;
  t2->SetBranchAddress("TLV branch",&v2);

  timer.Start();
  int n = (int) t2->GetEntries();
  std::cout << " Tree Entries " << n << std::endl; 
  etot = 0;
  for (int i = 0; i < n; ++i) { 
    t2->GetEntry(i);
    double Px = v2->Px();
    double Py = v2->Py();
    double Pz = v2->Pz();
    double E = v2->E();
    etot += E;
  }

  timer.Stop();
  std::cout << " Time for old Vector " << timer.RealTime() << "  " << timer.CpuTime() << endl; 
  std::cout << " E average" << etot/double(i) << endl; 
}


  
