#include <vector>
#include "TFile.h"
#include "TTree.h"
#include <iostream>
#include <list>
#include <set>
#include <map>

struct Inside {
   Inside() : fA(0),fB(0),fC(0) {}
   int fA;
   int fB;
   int fC;
};

class Values {
public:
   Values() : fX(0),fY(0) {}
   Values(int x, int y) : fX(x),fY(y) {}
   int fX;
   int fY;
   int fWillBeGone;
   Inside fGone;
   TObject subobj;
};

void Print(const Values &value)
{
   std::cout << "Single value\n";
   std::cout << " x=" << (int)value.fX;
   std::cout << " y=" << (int)value.fY;
   std::cout << "\n";
}

void Print(const std::vector<int> &fValues) {
   int i = 0;
   for(std::vector<int>::const_iterator iter = fValues.begin(); iter != fValues.end(); ++iter, ++i) {
      std::cout << " i=" << i;
      std::cout << " v=" << (int)(*iter);
      std::cout << "\n";
   }
}

void Print(const std::vector<Values> &fValues) {
   for(unsigned int i = 0; i < fValues.size(); ++i) {
      std::cout << " i=" << i;
      std::cout << " x=" << (int)fValues[i].fX;
      std::cout << " y=" << (int)fValues[i].fY;
      std::cout << "\n";
   }
}

void Print(const std::list<Values> &fValues) {
   int i = 0;
   for(std::list<Values>::const_iterator iter = fValues.begin(); iter != fValues.end(); ++iter, ++i) {
      std::cout << " i=" << i;
      std::cout << " x=" << (int)(*iter).fX;
      std::cout << " y=" << (int)(*iter).fY;
      std::cout << "\n";
   }
}

template <typename key, typename value>
void Print(const std::map<key,value> &fMap) {
   int i = 0;
   for(typename std::map<key,value>::const_iterator iter = fMap.begin(); iter != fMap.end(); ++iter, ++i) {
      std::cout << " i=" << i;
      std::cout << " key=" << iter->first;
      std::cout << " val=" << iter->second;
      std::cout << "\n";
   }
}

#ifdef __MAKECINT__
#pragma read sourceClass="Holder" targetClass="listHolder";
#endif

class listHolder {
public:
   int fX;
   Values fSingle;
   std::list<int>    fVec;
   std::list<Values> fValues;
   std::map<string,int> fMap;

   void Fill(int seed = 1) {
      fX = seed + seed/10.0;
      fSingle.fX = fSingle.fY = 99;
      for(int i = 0; i < 10*seed; ++i) {
         fVec.push_back(-i*0.1);
         fValues.push_back(Values(.1*seed+i,0.01*seed+i));
      }
      fMap["one"] = 99;
      fMap["two"] = -99;
   }

   void Print() const {
      std::cout << "listHolder:\n";
      std::cout << "listHolder::fX " << (int)fX << endl;
      ::Print(fSingle);
      std::cout << "listHolder::fVec :\n";
      int i = 0;
      for(std::list<int>::const_iterator iter = fVec.begin(); iter != fVec.end(); ++iter, ++i) {
         std::cout << " i=" << i;
         std::cout << " v=" << (int)(*iter);
         std::cout << "\n";
      }
      std::cout << "listHolder::fValues :\n";
      ::Print(fValues);
      std::cout << "listHolder::fMap :\n";
      ::Print(fMap);
   }
};

class Holder {
public:
   int    fX;
   Values fSingle;
   std::vector<int>    fVec;
   std::vector<Values> fValues;
   std::map<string,int> fMap;

   void Fill(int seed = 1) 
   {
      fX = seed;
      fSingle.fX = 10*seed+seed;
      fSingle.fY = 2*(10*seed+seed);
      fVec.clear();
      fValues.clear();
      for(int i = 0; i < 10*seed; ++i) {
         fVec.push_back(-i);
         fValues.push_back(Values(100*seed+i,1000*seed+i));
      }
      fMap["one"] = 100*seed;
      fMap["two"] = -200*seed;
  }

   void Print() const {
      std::cout << "Holder:\n";
      std::cout << "Holder::fX " << (int)fX << endl;
      ::Print(fSingle);
      std::cout << "Holder::fVec :\n";
      for(unsigned int i = 0; i < fVec.size(); ++i) {
         std::cout << " i=" << i;
         std::cout << " v=" << (int)fVec[i];
         std::cout << "\n";
      }
      std::cout << "Holder::fValues :\n";
      ::Print(fValues);
      std::cout << "Holder::fMap :\n";
      ::Print(fMap);
   }

};

void writeFile(const char * filename = "stlinnerevo.root") {
   TFile *f = TFile::Open(filename,"RECREATE");
   Values val(7,9);
   f->WriteObject(&val,"val");
   Holder h;
   h.Fill();
   h.Print();
   f->WriteObject(&(h.fVec),"vecint");
   f->WriteObject(&(h.fValues),"vec");
   f->WriteObject(&(h.fMap),"dict");
   f->WriteObject(&h,"holder");
   TTree *t = new TTree("T","T");
   t->Branch("holder",&h);
   t->Branch("vec",&h.fValues);
   t->Branch("vecint",&h.fVec);
   t->Fill();
   h.Fill(2);
   t->Fill();
   f->Write();
   delete f;
}

void readFile(const char * filename = "stlinnerevo.root") {
   TFile *f = TFile::Open(filename,"READ");

   std::cout << "Reading a Vale.\n";
   Values *val = 0;
   f->GetObject("val",val);
   if (val) Print(*val);
   else std::cout << "Value not read\n";

   std::cout << "Reading a vector<int>.\n";  
   vector<int> *vecint = 0;
   f->GetObject("vecint",vecint);
   if (vecint) Print(*vecint);
   else std::cout << "vector of int not read\n";

   std::cout << "Reading a vector<Values>.\n";
   vector<Values> *vec = 0;
   f->GetObject("vec",vec);
   if (vec) Print(*vec);
   else std::cout << "vector not read\n";

   std::cout << "Reading a map<string,int>.\n";
   map<string,int> *dict = 0;
   f->GetObject("dict",dict);
   if (dict) Print(*dict);
   else std::cout << "map not read\n";

   std::cout << "Reading a Holder object.\n";
   Holder *h = 0;
   f->GetObject("holder",h);
   h->Print();
   delete h; h = 0;

   TTree *t; f->GetObject("T",t);
   std::vector<Values> *vecbr = 0;
   vector<int> *vecintbr = 0;
   t->SetBranchAddress("vecint",&vecintbr);
   t->SetBranchAddress("vec",&vecbr);
   t->SetBranchAddress("holder",&h);
  
   for(Long64_t e = 0; e < 2 ; ++e) {
      t->GetEntry(e);

      std::cout << "Reading from TTree entry #" << e << ".\n";
      std::cout << "Reading a vector<int>.\n";
      Print(*vecintbr);
      std::cout << "Reading a vector<Values>.\n";
      Print(*vecbr);
      std::cout << "Reading a Holder object.\n";
      h->Print();
   }

   delete f;
}

void readFileList(const char * filename = "stlinnerevo.root") {
   TFile *f = TFile::Open(filename,"READ");
   Values *val = 0;
   f->GetObject("val",val);
   list<Values> *vec = 0;
   f->GetObject("vec",vec);
   if (vec) Print(*vec);
   else std::cout << "collection of float/int not read\n";
   listHolder *h = 0;
   f->GetObject("holder",h);
   h->Print();
   delete h; h = 0;
   TTree *t; f->GetObject("T",t);
   t->SetBranchAddress("holder",&h);
   t->GetEntry(0);
   h->Print();
   t->GetEntry(1);
   h->Print();
   delete f;
}

