void ctrans(char *filename) 
{
  TFile *_file0 = TFile::Open(filename);
  from = T;
  f = new TFile("clone.root","RECREATE");
  to = from->CloneTree(-1,"fast");
  //to->Import(from); 
  f->Write();
  delete _file0; delete f;
  gSystem->Exec(Form("mv clone.root %s",filename));
}
