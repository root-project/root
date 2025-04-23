class SomeVector
{
public:
   double dx = 1;
   double dy = 2;
   double dz = 3;
   double h = 4;

   SomeVector() = default;

   SomeVector(double x, double y, double z, double inh) : dx(x), dy(y), dz(z), h(inh) {}
};

class Middle {
public:
  SomeVector val;
};

class VecHolder {
public:
  std::vector<Middle> vec;
};

class Holder {
public:
  Middle mid;
};

struct Momentum {
  SomeVector pp;
  double ee = 5;
};

struct StepPointMC {
  SomeVector position;
  SomeVector postPosition;
  Momentum   momentum;
  SomeVector postMomentum;
};

struct StepPointVector {
  std::vector<StepPointMC> vec;
};

#ifdef __ROOTCLING__
#pragma link C++ options=version(11)  class SomeVector+;
#pragma link C++ class Middle+;
#pragma link C++ class Momentum+;
#pragma link C++ class Holder+;
#pragma link C++ class VecHolder+;
#pragma link C++ class StepPointMC+;
#pragma link C++ class StepPointVector+;
#endif

#include "TTree.h"
#include "TFile.h"

const char *filename = "splitcont.root";
void writefile(int splitlevel = 9)
{
  TFile *file = TFile::Open(filename, "RECREATE");
  TTree t("T", "t");
  VecHolder vh;
  vh.vec.resize(2);
  t.Branch("vecholder.", &vh, 32000, splitlevel);
  Holder h;
  t.Branch("holder.", &h, 32000, splitlevel);
  Middle m;
  t.Branch("middle.", &m, 32000, splitlevel);
  StepPointVector step;
  StepPointMC mc;
  std::vector<StepPointMC> stepvec;
  mc.position = SomeVector{1.25, 2.5, 5, 1};
  mc.postPosition = SomeVector{2.5, 5, 10, 2};
  mc.momentum.pp = SomeVector{3.75, 7.5, 15, 3};
  mc.momentum.ee = 66;
  mc.postMomentum = SomeVector{5, 10, 20, 4};
  step.vec.push_back(mc);
  mc.postMomentum = SomeVector{15, 20, 30, 5};
  stepvec.push_back(mc);
  t.Branch("stepvec.", &step);
  t.Branch("nodot_stepvec", &step);
  t.Branch("directstepvec.", &stepvec);
  t.Branch("nodot_directstepvec", &stepvec);

  t.Fill();
  // should not be needed :(
  //file->WriteObject(&h.vec[0], "content");
  file->Write();
  delete file;
}

int execCreateFileFullObjRule()
{
  writefile();
  return 0;
}
