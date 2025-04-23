#include "TObject.h"

class wrapper;

class myclass {
public:
   myclass *myself;
   wrapper *indirect;
   int fValue;

   myclass() : myself(this),indirect(0),fValue(0) {}
   virtual ~myclass() {}
      
   void set();

   void verify();

   ClassDefOverride(myclass,1);
};

class wrapper {
public:
   myclass *fParent;
   int fIndex;

   wrapper(myclass *p = 0) : fParent(p),fIndex(0) {}
   virtual ~wrapper() {}

   ClassDefOverride(wrapper,1);
};

inline void myclass::set() {
   indirect = new wrapper(this);
}

inline void myclass::verify() {
   if (myself != this) {
      fprintf(stdout,"The myself data member is incorrect\n");
   }
   if (indirect == 0) {
      fprintf(stdout,"The indirect pointer is still null\n");
   } else {
      if (this != indirect->fParent) {
         if (indirect->fParent==0) {
            fprintf(stdout,"The indirect fParent member is still null\n");
         } else {
            fprintf(stdout,"The indirect fParent member is incorrect\n");
         }
      }
   }
}

#include "TFile.h"
#include "TTree.h"

const char *defname = "myclass.root";

void write(const char *filename = defname) 
{
   TFile f(filename,"RECREATE");
   myclass *m = new myclass();
   m->set();

   f.WriteObject(m,"obj");

   TTree *tree = new TTree("T","T");
   // tree->Branch("split_1.","myclass",&m,32000,-1);
   tree->Branch("split0.","myclass",&m,32000,0);
   tree->Branch("split1.",&m,32000,1);
   tree->Branch("split99.",&m,32000,99);

   tree->Fill();
   tree->Write();

   f.Close();
}

void readobj(TTree *tree, const char *where)
{
   myclass *m = 0;
   tree->SetBranchAddress(where,&m);
   tree->GetEntry(0);
   if (!m) {
      fprintf(stdout,"reading failed in %s\n",where);
   } else {
      fprintf(stdout,"Verifying %s\n",where);
      m->verify();
   }
   tree->ResetBranchAddresses();
}

void read(const char *filename = defname)
{
   TFile f(filename,"READ");
   myclass *m;
   f.GetObject("obj",m);
   m->verify();

   TTree *tree;
   f.GetObject("T",tree);
   tree->Print();

   readobj(tree,"split0.");
   readobj(tree,"split1.");
   readobj(tree,"split99.");
}

   





