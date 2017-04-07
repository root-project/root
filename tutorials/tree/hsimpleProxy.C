/// \file
/// \ingroup tutorial_tree
/// \notebook -nodraw
///
/// To use this file, generate hsimple.root:
/// ~~~ {.cpp}
///    root.exe -b -l -q hsimple.C
/// ~~~
/// and do
/// ~~~ {.cpp}
///    TFile *file = TFile::Open("hsimple.root");
///    TTree *ntuple ;  file->GetObject("ntuple",ntuple);
///    ntuple->Draw("hsimpleProxy.C+");
/// ~~~
///
/// \macro_code
///
/// \author Rene Brun

double hsimpleProxy() {
   return px;
}
