const auto nbytesref = 10429; // Was 14204;
const auto nbranchesref = 1;

int test_numberBranchesRead(const char *fileName){
  auto treeName = "TotemNtuple";

  TFile *f = TFile::Open(fileName);

  TTreeReader reader(treeName, f);
  TTreeReaderValue<double> vd(reader, "track_rp_3.y");
  while(reader.Next()){
     *vd;
  }

  auto t = reader.GetTree();
  auto nbytes = f->GetBytesRead();
  auto fp = t->GetReadCache(f);
  auto nbranches = fp->GetCachedBranches()->GetEntriesFast();
  auto correctNumberBranches = nbranches == nbranchesref;
  auto correctNumberBytes    = nbytes == nbytesref;
  if (! correctNumberBytes){
    cerr << "Wrong number of bytes: read " << nbytes << " expected " << nbytesref << endl;
    return 1;
  }
  if (! correctNumberBranches){
    cerr << "Wrong number of branches: read " << nbranches << " expected " << nbranchesref << endl;
    return 1;
  }
  return 0;

}
