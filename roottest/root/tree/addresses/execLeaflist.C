struct particle{
   Double_t pt[6];
   Double_t pt_res[6];
   Double_t eta[6];
   Double_t eta_res[6];
   Double_t phi[6];
   Double_t phi_res[6];
   Int_t    id[6];
};


void execLeaflist() {
   TFile *f = TFile::Open("leaflist.root");
   TTree *t = (TTree*)f->Get("VBF");
   t->Print();


   particle Particle;
   Double_t Zon_mass;
   //cout << "Bad Address is " << (void*)&Particle << endl;
   //cout << "Good Address is " << (void*)&(Particle.pt[0]) << endl;

   cout << "First way of setting the addresses\n";
   t->SetBranchAddress("Particle",&(Particle.pt[0]));
   t->SetBranchAddress("Zon_mass",&Zon_mass);


   for(int i=0; i<10; i++){
      t->GetEntry(i);
      cout<< "Particle: " << Particle.pt[0] << " |  " << Particle.eta[0] << " |  " << Particle.phi[0] << endl;
      cout<< "Zon_mass: " << Zon_mass << endl;
   }

   
   cout << "Second way of setting the addresses\n";
   t->SetBranchAddress("Particle",Particle.pt);
   t->SetBranchAddress("Zon_mass",&Zon_mass);


   for(int i=0; i<10; i++){
      t->GetEntry(i);
      cout<< "Particle: " << Particle.pt[0] << " |  " << Particle.eta[0] << " |  " << Particle.phi[0] << endl;
      cout<< "Zon_mass: " << Zon_mass << endl;
   }
   
   
   cout << "Third way of setting the addresses\n";
   t->SetBranchAddress("Particle",(void*)&(Particle));
   t->SetBranchAddress("Zon_mass",&Zon_mass);


   for(int i=0; i<10; i++){
      t->GetEntry(i);
      cout<< "Particle: " << Particle.pt[0] << " |  " << Particle.eta[0] << " |  " << Particle.phi[0] << endl;
      cout<< "Zon_mass: " << Zon_mass << endl;
   }


   cout << "Fourth way of setting the addresses\n";
   t->SetBranchAddress("Particle",&(Particle));
   t->SetBranchAddress("Zon_mass",&Zon_mass);

   for(int i=0; i<10; i++){
      t->GetEntry(i);
      cout<< "Particle: " << Particle.pt[0] << " |  " << Particle.eta[0] << " |  " << Particle.phi[0] << endl;
      cout<< "Zon_mass: " << Zon_mass << endl;
   }
}