#include "myclass.h"

// required when macro compiled on Windows to correctly generate dictionary entries
#ifdef __ROOTCLING__
#pragma link C++ class myclass+;
#pragma link C++ class wrapper+;
#endif


#include "TFile.h"
#include "TTree.h"

const char *defname = "myclass.root";

void write(const char *filename = defname)
{
   TFile f(filename,"RECREATE");
   // avoid compression variation on different platforms
   f.SetCompressionLevel(0);
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

void verify_myclass()
{
   myclass *m = new myclass();
   m->verify();
   m->set();
   m->verify();
   delete m;
}

int verify()
{

   verify_myclass();

   write();

   read();

   return 0;
}

