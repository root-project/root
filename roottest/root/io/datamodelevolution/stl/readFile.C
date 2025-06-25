#include <vector>
#include "TFile.h"
#include "TTree.h"
#include <iostream>
#include <list>
#include <set>

#if 1

class Values {
public:
   Values() : fX(0),fY(0),fZ(0) {}
   Values(int x, int y) : fX(x),fY(y),fZ(0) {}
   int fX;
   int fY;
   int fZ;
};
#endif

class Track {
public:
   Track() : fX(0),fY(0) {}
   Track(float x, float y) : fX(x),fY(y) {}
   float fX;
   float fY;

#if 1
   int fWillBeGone;
   // Inside fGone;
   TObject subobj;
#endif

   bool operator<(const Track &right) const { return fX > right.fX; }
};

#if 0
typedef Track Values;
#endif

#ifdef __MAKECINT__
#pragma read sourceClass="Values" targetClass="Track";
#pragma link C++ class vector<Track>+;
#pragma read sourceClass="Holder" targetClass="listHolder";
#pragma read sourceClass="Holder" targetClass="setHolder";
#endif

void Print(const Track &value)
{
   std::cout << "Single value\n";
   std::cout << " x=" << (int)value.fX;
   std::cout << " y=" << (int)value.fY;
   std::cout << "\n";
}

void Print(const std::vector<float> &fValues) {
   for(unsigned int i = 0; i < fValues.size(); ++i) {
      std::cout << " i=" << i;
      std::cout << " v=" << (int)fValues[i];
      std::cout << "\n";
   }
}

void Print(const std::list<float> &fValues) {
   int i = 0;
   for(std::list<float>::const_iterator iter = fValues.begin(); iter != fValues.end(); ++iter, ++i) {
      std::cout << " i=" << i;
      std::cout << " v=" << (int)(*iter);
      std::cout << "\n";
   }
}

void Print(const std::set<float> &fValues) {
   int i = 0;
   for(std::set<float>::const_iterator iter = fValues.begin(); iter != fValues.end(); ++iter, ++i) {
      std::cout << " i=" << i;
      std::cout << " v=" << (int)(*iter);
      std::cout << "\n";
   }
}

void Print(const std::vector<Track> &fValues) {
   for(unsigned int i = 0; i < fValues.size(); ++i) {
      std::cout << " i=" << i;
      std::cout << " x=" << (int)fValues[i].fX;
      std::cout << " y=" << (int)fValues[i].fY;
      std::cout << "\n";
   }
}

void Print(const std::list<Track> &fValues) {
   int i = 0;
   for(std::list<Track>::const_iterator iter = fValues.begin(); iter != fValues.end(); ++iter, ++i) {
      std::cout << " i=" << i;
      std::cout << " x=" << (int)(*iter).fX;
      std::cout << " y=" << (int)(*iter).fY;
      std::cout << "\n";
   }
}

void Print(const std::set<Track> &fValues) {
   int i = 0;
   for(std::set<Track>::const_iterator iter = fValues.begin(); iter != fValues.end(); ++iter, ++i) {
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

class Holder {
public:
   float fX;
   Track fSingle;
   std::vector<float> fVec;
   std::vector<Track> fValues;
   std::map<string,long> fMap;

   void Fill(int seed = 1) {
      fX = seed + seed/10.0;
      fSingle.fX = fSingle.fY = 99.9;
      for(int i = 0; i < 10*seed; ++i) {
         fVec.push_back(-i*0.1);
         fValues.push_back(Track(.1*seed+i,0.01*seed+i));
      }
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
      std::cout << "listHolder::fMap :\n";
      ::Print(fMap);
   }
};

class listHolder {
public:
   float fX;
   Track fSingle;
   std::list<float>   fVec;
   std::list<Track> fValues;
   std::map<string,long> fMap;

   void Fill(int seed = 1) {
      fX = seed + seed/10.0;
      fSingle.fX = fSingle.fY = 99.9;
      for(int i = 0; i < 10*seed; ++i) {
         fVec.push_back(-i*0.1);
         fValues.push_back(Track(.1*seed+i,0.01*seed+i));
      }
   }

   void Print() const {
      std::cout << "listHolder:\n";
      std::cout << "listHolder::fX " << (int)fX << endl;
      ::Print(fSingle);
      std::cout << "listHolder::fVec :\n";
      int i = 0;
      for(std::list<float>::const_iterator iter = fVec.begin(); iter != fVec.end(); ++iter, ++i) {
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

class setHolder {
public:
   float fX;
   Track fSingle;
   std::set<float>   fVec;
   std::set<Track> fValues;
   std::map<string,long> fMap;

   void Fill(int seed = 1) {
      fX = seed + seed/10.0;
      fSingle.fX = fSingle.fY = 99.9;
      for(int i = 0; i < 10*seed; ++i) {
         fVec.insert(-i*0.1);
         fValues.insert(Track(.1*seed+i,0.01*seed+i));
      }
   }

   void Print() const {
      std::cout << "setHolder:\n";
      std::cout << "setHolder::fX " << (int)fX << endl;
      ::Print(fSingle);
      std::cout << "setHolder::fVec :\n";
      int i = 0;
      for(std::set<float>::const_iterator iter = fVec.begin(); iter != fVec.end(); ++iter, ++i) {
         std::cout << " i=" << i;
         std::cout << " v=" << (int)(*iter);
         std::cout << "\n";
      }
      std::cout << "setHolder::fValues :\n";
      ::Print(fValues);
      std::cout << "setHolder::fMap :\n";
      ::Print(fMap);
   }
};

void readFile(const char * filename = "stlinnerevo.root") {
   TFile *f = TFile::Open(filename,"READ");

   std::cout << "Reading a Track.\n";
   Track *tr = 0;
   f->GetObject("val",tr);
   if (tr) Print(*tr);
   else std::cout << "Value not read\n";

   std::cout << "Reading a vector<int> into a vector<float>.\n";
   vector<float> *vecint = 0;
   f->GetObject("vecint",vecint);
   if (vecint) Print(*vecint);
   else std::cout << "vector of float/int not read\n";

   std::cout << "Reading a vector<Values> into a vector<Track>.\n";
   vector<Track> *vec = 0;
   f->GetObject("vec",vec);
   if (vec) Print(*vec);
   else std::cout << "vector not read\n";

   std::cout << "Reading a map<string,int> into a map<string,long>.\n";
   map<string,long> *dict = 0;
   f->GetObject("dict",dict);
   if (dict) Print(*dict);
   else std::cout << "map not read\n";

   if (1) {
      std::cout << "Reading a Holder object.\n";
      Holder *h = 0;
      f->GetObject("holder",h);
      if(h) h->Print();
      delete h; h = 0;
   }
   if (1) {
      TTree *t; f->GetObject("T",t);
      Holder *h = 0;
      std::vector<Track> *vecbr = 0;
      vector<float> *vecintbr = 0;

      t->SetBranchAddress("vec",&vecbr);

      // t->GetBranch("vec")->GetEntry(0);
      // Print(*vecbr);
      // return;

      t->SetBranchAddress("vecint",&vecintbr);
      t->SetBranchAddress("vec",&vecbr);
      t->SetBranchAddress("holder",&h);
      for(Long64_t e = 0; e < 2 ; ++e) {
         t->GetEntry(e);

         std::cout << "Reading from TTree entry #" << e << ".\n";
         std::cout << "Reading a vector<int> into a vector<float>.\n";
         if (vecintbr) Print(*vecintbr);
         std::cout << "Reading a vector<Values> into a vector<Track>.\n";
         if (vecbr) Print(*vecbr);
         std::cout << "Reading a Holder object.\n";
         if (h) h->Print();
      }
      delete h;
   }
   delete f;
}

void readFileList(const char * filename = "stlinnerevo.root") {
   TFile *f = TFile::Open(filename,"READ");

   std::cout << "Reading a Track.\n";
   Track *tr = 0;
   f->GetObject("val",tr);
   if (tr) Print(*tr);
   else std::cout << "Value not read\n";

   std::cout << "Reading a vector<int> into a list<float>.\n";
   list<float> *vecint = 0;
   f->GetObject("vecint",vecint);
   if (vecint) Print(*vecint);
   else std::cout << "vector of float/int not read\n";

   std::cout << "Reading a vector<Values> into a list<Track>.\n";
   list<Track> *vec = 0;
   f->GetObject("vec",vec);
   if (vec) Print(*vec);
   else std::cout << "vector not read\n";

   std::cout << "Reading a map<string,int> into a map<string,long>.\n";
   map<string,long> *dict = 0;
   f->GetObject("dict",dict);
   if (dict) Print(*dict);
   else std::cout << "map not read\n";

   if (1) {
      std::cout << "Reading a listHolder object.\n";
      listHolder *h = 0;
      f->GetObject("holder",h);
      if (h) h->Print();
      delete h; h = 0;
   }
   if (1) {
      TTree *t; f->GetObject("T",t);
      listHolder *h = 0;
      std::list<Track> *vecbr = 0;
      list<float> *vecintbr = 0;

      // t->SetBranchAddress("holder",&h);

      // t->GetBranch("holder")->GetEntry(0);
      // h->Print();
      // return;

      t->SetBranchAddress("vecint",&vecintbr);
      t->SetBranchAddress("vec",&vecbr);
      t->SetBranchAddress("holder",&h);
      for(Long64_t e = 0; e < 2 ; ++e) {
         t->GetEntry(e);

         std::cout << "Reading from TTree entry #" << e << ".\n";
         std::cout << "Reading a vector<int> into a list<float>.\n";
         if (vecintbr) Print(*vecintbr);
         std::cout << "Reading a vector<Values> into a list<Track>.\n";
         if (vecbr) Print(*vecbr);
         std::cout << "Reading a listHolder object.\n";
         if (h) h->Print();
      }
   }
   delete f;
}

void readFileSet(const char * filename = "stlinnerevo.root") {
   TFile *f = TFile::Open(filename,"READ");

   std::cout << "Reading a Track.\n";
   Track *tr = 0;
   f->GetObject("val",tr);
   if (tr) Print(*tr);
   else std::cout << "Value not read\n";

   std::cout << "Reading a vector<int> into a set<float>.\n";
   set<float> *vecint = 0;
   f->GetObject("vecint",vecint);
   if (vecint) Print(*vecint);
   else std::cout << "vector of float/int not read\n";

   std::cout << "Reading a vector<Values> into a set<Track>.\n";
   set<Track> *vec = 0;
   f->GetObject("vec",vec);
   if (vec) Print(*vec);
   else std::cout << "vector not read\n";

   std::cout << "Reading a map<string,int> into a map<string,long>.\n";
   map<string,long> *dict = 0;
   f->GetObject("dict",dict);
   if (dict) Print(*dict);
   else std::cout << "map not read\n";

   if (1) {
      std::cout << "Reading a setHolder object.\n";
      setHolder *h = 0;
      f->GetObject("holder",h);
      if (h) h->Print();
      delete h; h = 0;
   }
   if (1) {
      TTree *t; f->GetObject("T",t);
      setHolder *h = 0;
      std::set<Track> *vecbr = 0;
      set<float> *vecintbr = 0;

      // t->SetBranchAddress("holder",&h);

      // t->GetBranch("holder")->GetEntry(0);
      // h->Print();
      // return;

      t->SetBranchAddress("vecint",&vecintbr);
      t->SetBranchAddress("vec",&vecbr);
      t->SetBranchAddress("holder",&h);
      for(Long64_t e = 0; e < 2 ; ++e) {
         t->GetEntry(e);

         std::cout << "Reading from TTree entry #" << e << ".\n";
         std::cout << "Reading a vector<int> into a set<float>.\n";
         if (vecintbr) Print(*vecintbr);
         std::cout << "Reading a vector<Values> into a set<Track>.\n";
         if (vecbr) Print(*vecbr);
         std::cout << "Reading a setHolder object.\n";
         if (h) h->Print();
      }
   }
   delete f;
}
