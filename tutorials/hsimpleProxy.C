// To use this file, generate hsimple.root:
//    root.exe -b -l -q hsimple.C
// and do
//    TFile *file = TFile::Open("hsimple.root");
//    TTree *ntuple = file->GetObject("ntuple",ntuple);
//    ntuple->Draw("hsimpleProxy.C+");
//
double hsimpleProxy() {
   return px;
}