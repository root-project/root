#include <vector>

struct DataBase {
   DataBase() : fId(0) {}
   int fId;
};

struct Data : public DataBase {
   Data() : fPx(0),fPy(0),fPz(0) {}
   float fPx;
   float fPy;
   float fPz;
};

struct Holder {
   std::vector<Data> fData;
};

#include "TFile.h"
#include "TTree.h"

void write(const char *filename = "foreignVec.root")
{
   Holder h;
#if 1
   Data d;
   h.fData.push_back(d);
   d.fId = 1;
   d.fPx = 2;
   d.fPy = 3;
   d.fPz = 4;
   h.fData.push_back(d);
   d.fId = 10;
   d.fPx = 11;
   d.fPy = 12;
   d.fPz = 13;
   h.fData.push_back(d);
#endif

   TFile f(filename,"RECREATE");
   f.WriteObject(&h,"h");

#if 1
   TTree tree("tree","title");
   tree.Branch("unsplit.",&h,32000,0);
   tree.Branch("split1.",&h,32000,1);
   tree.Branch("split9.",&h,32000,99);
   tree.Fill();
#endif
   f.Write();
   f.Close();
}

int Verify(Holder *h)
{
   if (h->fData.size() != 3) {
      printf("Error: size is not as expected, it is %d rather than 3.\n",(int)h->fData.size());
      return 1;
   }
   Data d = h->fData[2];
   if (d.fId != 10) {
      printf("Error: id is not as expected, it is %d rather than 10.\n",(int)d.fId);
      return 2;
   }
   if (d.fPx != 11) {
      printf("Error: fPx is not as expected, it is %d rather than 11.\n",(int)d.fPx);
      return 3;
   }
   if (d.fPy != 12) {
      printf("Error: fPx is not as expected, it is %d rather than 11.\n",(int)d.fPx);
      return 4;
   }
   if (d.fPz != 13) {
      printf("Error: fPx is not as expected, it is %d rather than 11.\n",(int)d.fPx);
      return 5;
   }
   return 0;
}

int VerifyBranch(TTree *tree, const char *bname) {
   Holder *h = 0;

   tree->SetBranchAddress(bname,&h);
   TBranch *b= tree->GetBranch(bname);
   b->GetEntry(0);
   if (!h) {
      printf("Error: did not read h for %s\n",bname);
      return 6;
   }
   int result = Verify(h);
   tree->ResetBranchAddresses();
   return result;
}

int execForeignVec(const char *filename = "foreignVec.root")
{
   TFile f(filename,"READ");
   Holder *h = 0;
   f.GetObject("h",h);
   if (!h) {
      printf("Error: did not find h\n");
      return 10;
   }
   int result = Verify(h);
   TTree *tree;
   f.GetObject("tree",tree);
   if (!tree) {
      printf("Error: did not find tree\n");
      return 11;
   }
   result += VerifyBranch(tree,"unsplit.");
   result += VerifyBranch(tree,"split1.");
   result += VerifyBranch(tree,"split9.");
   return result;
}
