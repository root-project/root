namespace edm {}
void runursula() {
  TFile *_file0 = TFile::Open("cmsursula.root");
  TTree *Events; _file0->GetObject("Events",Events);
  Events->GetEntry(0);
}
