#include <vector>

class MyVec {
public:
  // double vecval;
  int fN;
  double fFix[4];
  double *fArray; //[fN]
};

class Param {
public:
  MyVec par_;
};

class TrackBase {
public:
   double chi2;
   Param par_;
};

class Track : public TrackBase {
public:
  int val;
};

class Wrapper {
public:
   std::vector<Track> fObj;
};

#include "TTree.h"

void rundeepnest() {
  Wrapper *w = new Wrapper;
  TTree *t = new TTree("T","T");
  t->Branch("tracks.",&w);
}
