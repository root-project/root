void ctrans(const char *filename)
{
  TFile *_file0 = TFile::Open(filename);
  gSystem->Load("libEvent.so");
  TTree *from = (TTree*) _file0->Get("T");
  auto f = new TFile("clone.root","RECREATE");
  from->CloneTree(-1,"fast");
  //to->Import(from);
  f->Write();
  delete _file0; delete f;
  gSystem->Exec(Form("mv clone.root %s",filename));
}
