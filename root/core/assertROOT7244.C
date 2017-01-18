int assertROOT7244() {
  TString pwd = gSystem->pwd();
  gSystem->AddIncludePath("-I\"" + pwd + "/subdir_ROOT7244\"");
  if (!gSystem->IsFileInIncludePath("TheFile.h")) {
    Error("assertROOT7244", "Cannot find subdir_ROOT7244/TheFile.h");
    exit(1);
  }
  return 0;
}
