void ctrans(const char *filename, const char *outfilename)
{
  TFile *_file0 = TFile::Open(filename);
  TTree *from = (TTree*) _file0->Get("T");
  auto f = new TFile(outfilename, "RECREATE");
  printf("Call CloneTree\n");
  from->CloneTree(-1,"fast");
  //to->Import(from);
  printf("Call CloneTree done\n");

  f->Write();
  printf("delete _file0\n");
  delete _file0;
  printf("delete f\n");
  delete f;
}
