#include <vector>
#include "TObject.h"
#include "TString.h"

class Content : public TObject {
public:
   TString fName;
   const char *GetName() const { return fName; }
   void SetName(const char *name) { fName = name; }
   
   ClassDef(Content,3);
};

template <class T>
class DataVectorTmplt {
public:
   std::vector<Content> fValues; //!

   void Fill(unsigned long seed) {
      Content obj;
      for(size_t i = 0; i < seed; ++i) {
         obj.SetName(TString::Format("name%lu_%lu",i,seed));
         fValues.push_back(obj);
      }
   }

   void Print() {
      for(size_t i = 0; i < fValues.size(); ++i) {
         printf("values: %lu / %lu : %s\n",
                i, fValues.size(), fValues[i].GetName());
      }
   }
};

#ifdef __MAKECINT__
#pragma link C++ class DataVectorTmplt<Content>+;
#pragma link C++ class vector<Content>+;
#endif

typedef DataVectorTmplt<Content> DataVector;

#include "TFile.h"
#include "TClass.h"
#include "TTree.h"

   // This breaks the test on windows.
#ifndef protected
#define protected public
#endif
#include "TVirtualCollectionProxy.h"

int execWriteCustomCollection() {
   TFile *file = TFile::Open("coll.root","RECREATE");
   if (!file) return 1;

   TClass *c = TClass::GetClass("DataVector");
   c->CopyCollectionProxy( * TClass::GetClass("vector<Content")->GetCollectionProxy() );

   // This breaks the test on windows.
   c->GetCollectionProxy()->fClass = c;
   
   DataVector v;
   v.Fill(3);
   printf("Writing\n");
   v.Print();
   file->WriteObject(&v,"coll");
   TTree *tree = new TTree("T","T");
   tree->Branch("coll.",&v);
   tree->Branch("vec.",&v.fValues);
   tree->Fill();
   file->Write();
   delete file;

   printf("Reading\n");
   DataVector *vp = 0;
   file = TFile::Open("coll.root","READ");
   file->GetObject("coll",vp);
   if (vp) {
      vp->Print();
   }
   printf("Reading TTree\n");
   file->GetObject("T",tree);
   if (tree) {
      DataVector *tvp = 0;
      tree->SetBranchAddress("coll",&tvp);
      tree->GetEntry(0);
      if (tvp) tvp->Print();
   }
   delete file;
   return 0;
}
