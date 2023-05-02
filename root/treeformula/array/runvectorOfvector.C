#include <vector>
#include "TROOT.h"
#include "TCanvas.h"

typedef std::vector<unsigned int> subUCollection;
typedef std::vector<subUCollection> Ucollection;
typedef std::vector<int> subCollection;
typedef std::vector<subCollection> collection;
#ifndef __CINT__
typedef std::vector<collection> third;
#endif

// classZ.vector<vector<int> >      -> 4, t->Draw("vec.fVec1");t->Draw("vec.fVecPtr");
// vector<classX.vector<classY> > > -> 5, t->Draw("vec.fVec2");
//                                        t->Draw("vec.fVec2.fN");
//                                        t->Draw("vec.fVec2.fVec1");
// vector<classX.vector<int> > >    -> 5, t->Draw("vec.fVec3");

#include "TObject.h"
struct One : public TObject {
   One() : fN(0),fVecPtr(new subCollection) {}
   One(const One &rhs) : TObject(),fN(rhs.fN),fVecPtr(new subCollection(*rhs.fVecPtr)) {}
   ~One() override { delete fVecPtr; }
   int GetN1() { return fN+1; }
   int fN;
   subCollection *fVecPtr; //->
   subUCollection fUVec;
   collection fVec1;
   ClassDefOverride(One,1);
};

struct Two {
   std::vector<One> fVec2;
   collection fVec3;
   subCollection fVec4;
};

typedef std::vector<Two> TwoCollec;

#include "TTree.h"

#ifdef __MAKECINT__
#pragma link C++ class std::vector<int>;
#pragma link C++ class std::vector<subCollection>;
#pragma link C++ class std::vector<One>;
#pragma link C++ class std::vector<Two>;
//#pragma link C++ class std::vector<collection>;

#endif

TTree* generateTree_1(int sub = 10) {
   TTree *t = new TTree("1","");

   subCollection *p = new subCollection;
   
   t->Branch("vec",&p);

   int total = 0;
   for(int i=0; i<sub; ++i) {
      p->push_back(3*(i+1));
      ++total;
   }
   t->Fill();
   t->ResetBranchAddresses();
   fprintf(stderr,"Total created %d\n",total);
   return t;
}

TTree* generateTree_3(int sub = 10) {
   TTree *t = new TTree("3","");

   third *p = new third;

   t->Branch("vec",&p);

   collection c;
   subCollection a;
   int total = 0;
   for(int i=0; i<sub; ++i) {
      a.clear();
      for(int j=0; j<3*(i+1); ++j) {
         a.push_back(j);
         ++total;
      }
      c.push_back(a);
   }
   p->push_back(c);
   t->Fill();
   t->ResetBranchAddresses();
   fprintf(stderr,"Total created %d\n",total);
   return t;
}

TTree* generateTree_4(int sub = 10) {
   TTree *t = new TTree("4","");

   One *p = new One;

   t->Branch("vec",&p,32000,0);

   subCollection a;
   p->fN = 0;
   for(int i=0; i<sub; ++i) {
      a.clear();
      for(int j=0; j<3*(i+1); ++j) {
         a.push_back(j);
         p->fVecPtr->push_back(j+1);
         p->fUVec.push_back(j+1);
         ++p->fN;
      }
      p->fVec1.push_back(a);
   }
   t->Fill();
   t->ResetBranchAddresses();
   fprintf(stderr,"Total created %d\n",p->fN);
   return t;
}

TTree* generateTree_5(int sub = 10) {
   TTree *t = new TTree("5","");

   TwoCollec *p = new TwoCollec;

   t->Branch("vec",&p,32000,0);

   Two two;
   One one;
   subCollection a;
   for(int i=0; i<sub; ++i) {
      a.clear();
      for(int j=0; j<3*(i+1); ++j) {
         two.fVec4.push_back(j);
         a.push_back(j);
         ++one.fN;
      }
      one.fVec1.push_back(a);
      two.fVec3.push_back(a);
   }
   two.fVec2.push_back(one);
   two.fVec2.push_back(one);
   p->push_back(two);
   two.fVec2.push_back(one);
   p->push_back(two);
   t->Fill();
   t->ResetBranchAddresses();
   fprintf(stderr,"Total created %d\n",one.fN*(2+3));
   return t;
}

TTree* generateTree_6(int sub = 10) {
   TTree *t = new TTree("6","");

   One *p = new One;

   t->Branch("vec",&p,32000,1);

   subCollection a;
   p->fN = 0;
   for(int i=0; i<sub; ++i) {
      a.clear();
      for(int j=0; j<3*(i+1); ++j) {
         a.push_back(j);
         p->fVecPtr->push_back(j+1);
         p->fUVec.push_back(j+1);
         ++p->fN;
      }
      p->fVec1.push_back(a);
   }
   t->Fill();
   t->ResetBranchAddresses();
   fprintf(stderr,"Total created %d\n",p->fN);
   return t;
}

TTree* generateTree_7(int sub = 10) {
   TTree *t = new TTree("7","");

   TwoCollec *p = new TwoCollec;

   t->Branch("vec",&p,32000,99);

   Two two;
   One one;
   subCollection a;
   for(int i=0; i<sub; ++i) {
      a.clear();
      for(int j=0; j<3*(i+1); ++j) {
         two.fVec4.push_back(j);
         a.push_back(j);
         ++one.fN;
      }
      one.fVec1.push_back(a);
      two.fVec3.push_back(a);
   }
   two.fVec2.push_back(one);
   two.fVec2.push_back(one);
   p->push_back(two);
   two.fVec2.push_back(one);
   p->push_back(two);
   t->Fill();
   t->ResetBranchAddresses();
   fprintf(stderr,"Total created %d\n",one.fN*(2+3));
   return t;
}

TTree* generateTree_8(int sub = 10) {
   TTree *t = new TTree("8","");

   One *p = new One;

   t->Branch("vec",&p,32000,-1);

   subCollection a;
   p->fN = 0;
   for(int i=0; i<sub; ++i) {
      a.clear();
      for(int j=0; j<3*(i+1); ++j) {
         a.push_back(j);
         p->fVecPtr->push_back(j+1);
         ++p->fN;
      }
      p->fVec1.push_back(a);
   }
   t->Fill();
   t->ResetBranchAddresses();
   fprintf(stderr,"Total created %d\n",p->fN);
   return t;
}

TTree* generateTree(int sub = 10, int level = 2) {
   switch(level) {
      case 1: return generateTree_1(sub);
      case 3: return generateTree_3(sub);
      case 4: return generateTree_4(sub);
      case 5: return generateTree_5(sub);
      case 6: return generateTree_6(sub);
      case 7: return generateTree_7(sub);
      case 8: return generateTree_8(sub);
   }
   TTree *t = new TTree("2","");

   collection *p = new collection;
   
   t->Branch("vec",&p);

   subCollection a;
   int total = 0;
   for(int i=0; i<sub; ++i) {
      a.clear();
      for(int j=0; j<3*(i+1); ++j) {
         a.push_back(j);
         ++total;
      }
      p->push_back(a);
   }
   t->Fill();
   t->ResetBranchAddresses();
   fprintf(stderr,"Total created %d\n",total);
   return t;
}

bool testing(TTree *t, const char *what, Int_t expect) 
{
   Int_t res = t->Draw(what,"","");
   if (res!=expect) {
      fprintf(stderr,"Error: %s t->Draw(\"%s\") drew %d instead of %d values\n",
              t->GetName(), what,res,expect);
      return false;
   }
   return true;
}

bool runvectorOfvector() {
   //gROOT->ProcessLine("SetROOTMessageToStdout();");
   //gROOT->ProcessLine("#include <vector>");

   bool success = true;


   /* TCanvas *c = */ new TCanvas("mycanvas");
   TTree *t;
   t = generateTree(10,1);
   success &= testing(t,"vec",10);

   t = generateTree(10,2);
   success &= testing(t,"vec",165);

   t = generateTree(10,3);
   success &= testing(t,"vec",-1);

   t = generateTree(10,4);
   success &= testing(t,"vec.fVec1",165);
   success &= testing(t,"vec.fVecPtr",165);
   success &= testing(t,"vec.fVec1[][0]",10);
   success &= testing(t,"vec.fVecPtr[][1]",165);
   success &= testing(t,"vec.fVec1[1][]",6);
   success &= testing(t,"vec.fVecPtr[2][]",1);

   t = generateTree(10,5);
   success &= testing(t,"vec.fVec2",0);
#ifndef ClingWorkAroundCallfuncAndInline
   success &= testing(t,"vec.fVec2.GetN1()",5);
   success &= testing(t,"vec.fVec2.GetN1()-vec.fVec2.fN",5);
#endif
   success &= testing(t,"vec.fVec2.fN",5);
   success &= testing(t,"vec.fVec2.fVec1",0);
   success &= testing(t,"vec.fVec3",0);
   success &= testing(t,"vec.fVec4",330);
   success &= testing(t,"vec.fVec1",0);

   t = generateTree(10,6);
   success &= testing(t,"vec.fVec1",165);
   success &= testing(t,"vec.fVecPtr",165);
   success &= testing(t,"vec.fVec1[][0]",10);
   success &= testing(t,"vec.fVecPtr[][1]",165);
   success &= testing(t,"vec.fVec1[1][]",6);
   success &= testing(t,"vec.fVecPtr[2][]",1);

   t = generateTree(10,7);
   success &= testing(t,"vec.fVec2",0);
#ifndef ClingWorkAroundCallfuncAndInline
   success &= testing(t,"vec.fVec2.GetN1()",5);
   success &= testing(t,"vec.fVec2.GetN1()-vec.fVec2.fN",5);
#endif
   success &= testing(t,"vec.fVec2.fN",5);
   success &= testing(t,"vec.fVec2.fVec1",0);
   success &= testing(t,"vec.fVec3",0);
   success &= testing(t,"vec.fVec4",330);
   success &= testing(t,"vec.fVec1",0);

   t = generateTree(10,8);
   success &= testing(t,"vec.fVec1",165);
   success &= testing(t,"vec.fVecPtr",165);
   success &= testing(t,"vec.fVec1[][0]",10);
   success &= testing(t,"vec.fVecPtr[][1]",165);
   success &= testing(t,"vec.fVec1[1][]",6);
   success &= testing(t,"vec.fVecPtr[2][]",1);

   fprintf(stderr,"The end\n");
   ::fflush(stderr);
   // To be consistent with the makefile we need to return true for failure
   return !success;
}


