class SomeVector
{
public:
   double data[3] = {-1, -1, -1};
   double h;

  SomeVector() = default;

  SomeVector(double x, double y, double z, double inh) : h(inh) {
    data[0] = x;
    data[1] = y;
    data[2] = z;
  }
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
  double ee = -1;
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
#pragma link C++ options=version(12)  class SomeVector+;
#pragma link C++ class Middle+;
#pragma link C++ class Momentum+;
#pragma link C++ class Holder+;
#pragma link C++ class VecHolder+;
#pragma link C++ class StepPointMC+;
#pragma link C++ class StepPointVector+;

#pragma read sourceClass="SomeVector" version="[-11]" \
        targetClass="SomeVector" \
        source="double dx; double dy; double dz;" \
        target="data" \
        code="{ data[0] = onfile.dx; data[1] = onfile.dy; data[2] = onfile.dz; }";

#pragma read sourceClass="SomeVector1" version="[-11]" \
        targetClass="SomeVector" \
        source="double dx; double dy; double dz;" \
        target="" \
        code="{ newObj->data[0] = onfile.dx; newObj->data[1] = onfile.dy; newObj->data[2] = onfile.dz; cout << (void*)onfile_add << ' ' << onfile.dx << endl; }";

#endif

#include "TTree.h"
#include "TFile.h"
#include <iostream>
#include <iomanip>
#include "TSystem.h"

//void Print(const StepPointMC &s)

std::ostream &format(std::ostream &os)
{
  os << std::fixed;
  os << std::setw(5);
  os << std::setprecision(2);
  return os;
}

std::ostream &operator<<(std::ostream &os, const StepPointMC &s)
{
    os << std::setw(5);
    os << "position:     {";
    for(auto &d : s.position.data)
      format(os) << d << ", ";
    format(os);
    os << s.position.h;
    os << "}\npostPosition: {";
    for(auto &d : s.postPosition.data)
      format(os) << d << ", ";
    format(os) << s.postPosition.h;
    os << "}\nmomentum:     {{";
    for(auto &d : s.momentum.pp.data)
      format(os) << d << ", ";
    format(os) << s.momentum.pp.h;
    os << "}, ";
    format(os) << s.momentum.ee;
    os << "}\npostMomentum: {";
    for(auto &d : s.postMomentum.data)
      format(os) << d << ", ";
    format(os) << s.postMomentum.h;
    os << "}\n";
    return os;
}

int compare(double read, double expected)
{
  return ( (read - expected) > 1e-6 );
}

int check(const SomeVector &obj, std::vector<double> values)
{
  int result = compare(obj.data[0] , values[0]);
  result = compare(obj.data[1] , values[1]) + result;
  result = compare(obj.data[2] , values[2]) + result;
  result = compare(obj.h , values[3]) + result;
  //fprintf(stderr, "result = %d %g %g %g %g\n", result, obj.data[0], obj.data[1], obj.data[2], obj.h);
  //fprintf(stderr, "result = %d %g %g %g %g\n", result, values[0], values[1], values[2], values[3]);
  return result;
}

int check(const StepPointMC &step, std::vector<std::vector<double>> values)
{
  int result = check(step.position, values[0]);
  result = check(step.position, values[1]) + result;
  result = check(step.momentum.pp, values[2]) + result;
  result = check(step.postMomentum, values[3]) + result;
  //fprintf(stderr, "result = %d\n", result);
  return result;
}

const char *filename = "splitcont.root";
bool readfile()
{
  TFile *file = TFile::Open(filename, "READ");
  TTree *tree = file->Get<TTree>("T");
  Holder *h = nullptr;
  VecHolder *vh = nullptr;
  Middle *m = nullptr;
  StepPointVector *step = nullptr;
  StepPointVector *nodotstep = nullptr;
  std::vector<StepPointMC> *stepvec = nullptr;
  std::vector<StepPointMC> *nodotstepvec = nullptr;

  tree->SetBranchAddress("holder.", &h);
  tree->SetBranchAddress("vecholder.", &vh);
  tree->SetBranchAddress("middle.", &m);
  tree->SetBranchAddress("stepvec.", &step);
  tree->SetBranchAddress("nodot_stepvec", &nodotstep);
  // This is broken, we would need to pass the name without a dot!
  // tree->SetBranchAddress("directstepvec.", &stepvec);
  tree->SetBranchAddress("directstepvec", &stepvec);
  tree->SetBranchAddress("nodot_directstepvec", &nodotstepvec);

  tree->GetEntry(0);

  int result = false;

  std::cout << "Read directly: " << '\n';
  for(auto &d : m->val.data)
    std::cout << " d value : " << d << '\n';
  std::cout << " h value : " << m->val.h << '\n';
  result = check(m->val, {1, 2, 3, 4}) + result;

  std::cout << "Read Object: " << '\n';
  for(auto &d : h->mid.val.data)
    std::cout << " d value : " << d << '\n';
  std::cout << " h value : " << h->mid.val.h << '\n';
  result = check(h->mid.val, {1, 2, 3, 4}) + result;

  std::cout << "Read Vector: " << '\n';
  for(auto &obj : vh->vec) {
    for(auto &d : obj.val.data)
      std::cout << " d value : " << d << '\n';
    std::cout << " h value : " << obj.val.h << '\n';
    result = check(obj.val, {1, 2, 3, 4}) + result;
  }

  std::cout << "Read SetPointMC vector: \n";
  for(auto &s : step->vec) {
    result = check(s, 
      {{ 1.25,  2.50,  5.00,  1.00},
       { 2.50,  5.00, 10.00,  2.00},
       { 3.75,  7.50, 15.00,  3.00},
       { 5.00, 10.00, 20.00,  4.00}}
    ) + result;
    std::cout << s;
  }
  std::cout << "Read SetPointMC vector without dot: \n";
  for(auto &s : nodotstep->vec) {
    result = check(s, 
      {{ 1.25,  2.50,  5.00,  1.00},
       { 2.50,  5.00, 10.00,  2.00},
       { 3.75,  7.50, 15.00,  3.00},
       { 5.00, 10.00, 20.00,  4.00}}
      ) + result;
    std::cout << s;
  }
  std::cout << "Read SetPointMC direct vector: \n";
  for(auto &s : *stepvec) {
    result = check(s, 
      {{  1.25,  2.50,  5.00,  1.00},
       {  2.50,  5.00, 10.00,  2.00},
       {  3.75,  7.50, 15.00,  3.00},
       { 15.00, 20.00, 30.00,  5.00}}
      ) + result;
     std::cout << s;
  }

  // Not working because the address of the sub-branch is not set (fObject == nullptr)
  std::cout << "Read SetPointMC direct vector without dot: \n";
  for(auto &s : *nodotstepvec) {
    result = check(s, 
      {{  1.25,  2.50,  5.00,  1.00},
       {  2.50,  5.00, 10.00,  2.00},
       {  3.75,  7.50, 15.00,  3.00},
       { 15.00, 20.00, 30.00,  5.00}}
      ) + result;
    std::cout << s;
  }

  delete file;
  //fprintf(stderr, "result = %d\n", result);
  if (result)
    gSystem->Exit(result);
  return result;
}

int execReadFileFullObjRule() {
  return readfile();
}

