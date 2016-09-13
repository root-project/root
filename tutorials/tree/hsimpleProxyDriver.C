/// \file
/// \ingroup tutorial_tree
/// \notebook -nodraw
/// This is the driver of the hsimpleProxy example
/// It provides the infrastructure to run that code on an ntuple
/// To be run from the tutorials directory
///
/// \macro_code
///
/// \author Rene Brun

void hsimpleProxyDriver()
{
   std::cout << gSystem->WorkingDirectory() << std::endl;
   TFile *file = TFile::Open("hsimple.root");
   if (!file){
      std::cerr << "Input file not found.\n";
      return ;
   }
   TTree *ntuple = nullptr;
   file->GetObject("ntuple",ntuple);
   TString dir = gSystem->DirName(__FILE__);
   ntuple->Draw(dir+"/hsimpleProxy.C+");
}
