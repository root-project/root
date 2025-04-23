class Top {
public:
   int i;
   virtual ~Top() {}
};

class AbstractMiddle :public Top {
public:
   
   virtual int get() = 0;
};

class Bottom : public AbstractMiddle {
public:
   int j;
   int get() override { return j; }
};

class Holder {
public:
   Holder(): b(0) {}
   Top *b;
};

#include "TFile.h"
#include "TTree.h"

const char *defaultname = "abstract.root";

void writefile(const char *filename = defaultname) 
{
   TFile f(filename,"RECREATE");
   TTree * t = new TTree("tree","tree");
   Holder b;
   b.b = new Bottom;
   t->Branch("obj.",&b,32000,0);
   t->Fill();
   f.Write();
}

void clonefile(const char *filename = defaultname) 
{
   TString copyname("copy-");
   copyname.Append(filename);
   
   TFile f(filename);
   TTree *in; f.GetObject("tree",in);
   
   TFile copy(copyname,"RECREATE");
   in->CloneTree(-1,"fast");
   
   copy.Write();
}
   

void abstract(int mode) {
   switch(mode) {
     case 0: writefile(); break;
     case 1: clonefile(); break;
   }
}
