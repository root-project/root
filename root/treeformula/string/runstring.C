#include "TString.h"
#include "TTree.h"
#include "TFile.h"

#include <vector>
#include <string>

using namespace std;

class myclass {
public:
   TString sone;
   string stwo;
   
   vector<TString> sthree;
   vector<string> sfour;
   vector<bool> sfive;
   
   myclass() {}

   myclass(int i) {
      sone = "sone";
      stwo = "stwo";
      
      sthree.push_back("sthree_1");
      sthree.push_back("sthree_2");
      sfour.push_back("sfour_1");
      sfour.push_back("sfour_2");
      sfive.push_back(true);
      sfive.push_back(false);
      sfive.push_back(true);
      sfive.push_back(false);
   }
};

      


#ifdef __MAKECINT__
#pragma link C++ class vector<string>+;
#endif

void runstring() {
   TString *sone = new TString("sone");
   string *stwo = new string("stwo");

   vector<TString> *sthree = new vector<TString>;
   vector<string> *sfour = new vector<string>;
   vector<bool> sfive;
   vector<int> ssix;
   
   sthree->push_back("sthree_1");
   sthree->push_back("sthree_2");
   sfour->push_back("sfour_1");
   sfour->push_back("sfour_2");
   sfive.push_back(false);
   sfive.push_back(true);
   sfive.push_back(false);
   ssix.push_back(33);
   
   myclass *m = new myclass(1);

   TFile *f = new TFile("string.root","RECREATE");
   TTree *t = new TTree("T","T");

   //t->Branch("sone",&sone,32000,0);
   //t->Branch("stwo",&stwo,32000,0);
   t->Branch("sthree",&sthree,32000,0);
   t->Branch("sfour",&sfour,32000,0);
   t->Branch("obj.",&m);
   t->Branch("sfive",&sfive);
   t->Branch("ssix",&ssix);
 
   t->Fill();

   f->Write();
   
   t->Scan("*");
   t->Scan("sthree.Data()");
   t->Scan("sfour.c_str()");
}
