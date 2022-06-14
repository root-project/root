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
   std::string s1(__FILE__);
   TString dir = gSystem->UnixPathName(s1.substr(0, s1.find_last_of("\\/")).c_str());
   ntuple->Draw(dir+"/hsimpleProxy.C+");
}
