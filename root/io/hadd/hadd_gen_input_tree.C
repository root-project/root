#include <TTree.h>
#include <TFile.h>
#include <TSystem.h>

#include <memory>

void hadd_gen_input_tree(const char *fname)
{
   std::unique_ptr<TFile> file { TFile::Open(fname, "RECREATE") };
   file->SetCompressionSettings(101);
   TTree tree("t", "t");
   int x;
   tree.Branch("x", &x)->SetCompressionSettings(101);
   x = 42;
   tree.Fill();
   tree.Write();
}
