{
  Int_t a, b, c;

  TTree *t = new TTree( "aname", "atitle" );
  t->Branch( "var_a", &a, "var_a/I" );
  t->Branch( "var_b", &b, "var_b/I" );
  t->Branch( "var_c", &c, "var_c/I" );

  for( int i=0; i<10; i++ ){
     a = i;
     b = i*2;
     c = i*4;
     t->Fill();
  }

  TString filename = "memory.root";
  TFile *f = TFile::Open( filename, "RECREATE" );
  t->Write();
  f->Close();
  delete f;

  f = TFile::Open(filename);
  TTree *tree; f->GetObject("aname",tree);
  tree->Scan("*");
  delete f;
}
