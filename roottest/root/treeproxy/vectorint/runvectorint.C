#include <vector>
#include "TFile.h"
#include "TTree.h"
#include "TROOT.h"
#include "TClonesArray.h"

#ifndef runvectorint_C
#ifdef ClingWorkAroundMultipleInclude
#define runvectorint_C
#endif

class Track : public TObject {
public:
   int random;
   ClassDef(Track,1);
};

#ifdef __MAKECINT__
#pragma link C++ class vector<vector<double> >+;
//NOTYET #pragma link C++ class vector<vector<vector<double> > >+;
#pragma link C++ class Track+;
#endif

namespace std {} using namespace std;

class Top {
public:
   Top() : fTracks("Track") {
   }
  vector<double> vals;
  vector<vector<double> > vecvals;
  vector<Track> tracks;
  vector<Track*> ptrtracks;
  TClonesArray fTracks;
};


void createvec(const char *filename = "vec.root")
{
   gROOT->ProcessLine("#include <vector>");
   TFile *f = TFile::Open(filename,"RECREATE");
   TTree *t = new TTree("t","t");
   std::vector<double> *d = new std::vector<double>;
   d->push_back(3.0);
   d->push_back(6.0);
   std::vector<std::vector<double> > *dvec = new std::vector<std::vector<double> >;
   dvec->push_back(*d);
   std::vector<int> *i = new std::vector<int>;
   i->push_back(3);
   i->push_back(6);

#ifdef NOTYET
   std::vector<std::vector<std::vector<double> > > *tvec = new std::vector<std::vector<std::vector<double> > >;
   tvec->push_back( *dvec );
#endif

   Top *top = new Top;
   t->Branch("myvec.",&d);
   t->Branch("myvecvec.",&dvec);
   t->Branch("myint.someodd.name",&i);
#ifdef NOTYET
   t->Branch("myvecvecvec.",&tvec);
#endif
   t->Branch("topobj.",&top);
   t->Fill();
   f->Write();
}

void createsel(const char *filename = "vec.root")
{
   TFile *f = TFile::Open(filename,"READ");
   TTree *t; f->GetObject("t",t);
   t->MakeProxy("vectorintSel","dude.C","","");
}

void usesel(const char *filename = "vec.root")
{
   TFile *f = TFile::Open(filename,"READ");
   TTree *t; f->GetObject("t",t); 
   t->Process("vectorintSel.h+","goff");
}

int runvectorint(int mode = 0) 
{
   if (mode==0) {
     createvec();
     createsel();
     usesel();
   } else if (mode==1) {
     createvec();
     createsel();
   } else if (mode==2) {
     createsel();
     usesel();
   } else if (mode==3) {
     createvec();
   } else if (mode==4) {
     createsel();
   } else if (mode==5) {
     usesel();
   } else {
     return 1;
   }
   return 0;
}

#endif
