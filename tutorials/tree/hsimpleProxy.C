/// \file
/// \ingroup tutorial_tree
/// \notebook -nodraw
/// Used by hsimpleProxyDriver.C.
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
/// \macro_image (tcanvas_js)
/// \preview 
/// \macro_code
///
/// \date October 2024
/// \author Rene Brun

double hsimpleProxy() {
   return px;
}
