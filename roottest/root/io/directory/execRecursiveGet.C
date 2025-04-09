{
   TFile *file = TFile::Open("objstring.root");
   file->Get("Lumi/physics;1")->Print();
   file->Get("Lumi/physics;2")->Print();
}
