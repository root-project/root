int RootcpReplaceEntireFileCheck(const char *fname) {
  auto file = TFile::Open(fname);
  int retval = file->Get<TH1D>("hpx") == nullptr;
  delete file;
  // Returns 0 on success (file->Get was successful), 1 otherwise.
  return retval;
}
