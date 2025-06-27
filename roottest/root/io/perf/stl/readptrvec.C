#include "TLorentzVector.h"
#include <vector>

#ifdef __ROOTCLING__
#pragma link C++ class vector<TLorentzVector>+;
#endif

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

class Holder {

private:
   std::vector<TLorentzVector*> fLorentzVec;
   std::vector<int> fIntVec;

public:
   ~Holder() {
      for(unsigned int j = 0; j < fLorentzVec.size(); ++j) {
         delete fLorentzVec[j];
      }
   }
   void Fill(int vsize) {
      for(int j = 0; j < vsize ; ++j) {
         fIntVec.push_back(j);
         fLorentzVec.push_back(new TLorentzVector(j,j+1,j+2,j+3));
      }
   }
};

void writefile(const char *filename = "ptrvec.root", int vsize = 2000, int tsize = 1000) {
   TFile f(filename,"RECREATE");

   Holder holder;
   holder.Fill(vsize);

   f.WriteObject(&holder,"obj");

   TTree tree("tree","tree");
   tree.Branch("split.",&holder,32000,99);
   tree.Branch("strm.",&holder,32000,0);

   for(int i = 0; i < tsize; ++i) {
      tree.Fill();
   }
   f.Write();
}

void readobj(TFile *file)
{
   Holder *holder;
   file->GetObject("obj",holder);
   delete holder;
}

TTree *readtree(TFile *file)
{
   TTree *tree;
   file->GetObject("tree",tree);
   return tree;
}

Long64_t readsplit(TTree *tree)
{
   TBranch *branch = tree->GetBranch("split.");
   Long64_t entries = tree->GetEntries();
   Long64_t nbytes = 0;
   for(Long64_t i = 0; i < entries; ++i) {
      nbytes += branch->GetEntry(i);
   }
   return nbytes;
}

Long64_t readstrm(TTree *tree)
{
   TBranch *branch = tree->GetBranch("strm.");
   Long64_t entries = tree->GetEntries();
   Long64_t nbytes = 0;
   for(Long64_t i = 0; i < entries; ++i) {
      nbytes += branch->GetEntry(i);
   }
   return nbytes;
}

void readfile(int what = 7, const char *filename = "ptrvec.root") {
   // Set bit in 'what' to request part of the reading:
   //   1 : the object
   //   2 : the split branch
   //   4 : the streamed branch

   TFile *file = TFile::Open(filename);

   if (what & 1) {
      readobj(file);
   }
   TTree *tree = readtree(file);

   if (what & 2) {
      readsplit(tree);
   }
   if (what & 4) {
      readstrm(tree);
   }
}

void readptrvec(int what = 7, const char *filename = "ptrvec.root")
{
   writefile(filename);

   readfile(what, filename);
}


