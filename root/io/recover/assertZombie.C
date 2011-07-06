{
  TFile *_file0 = TFile::Open("crashed.root");
  if (_file0 == 0) {
    return 0;
  } else {
    return 1;
  }
}
