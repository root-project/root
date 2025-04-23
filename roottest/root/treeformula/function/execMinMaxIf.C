#include "TTree.h"
#include "TTreeFormula.h"
#include <vector>

int execMinMaxIf() {
  TTree t("tree","tree");
  std::vector<int>* b = new std::vector<int>;
  t.Branch("b",&b);
  b->push_back(4);
  t.Fill();

  TTreeFormula fMin("min_test","MinIf$(b,Iteration$==Length$-1)",&t);
  fMin.GetNdata();
  TTreeFormula fMax("max_test","MaxIf$(b,Iteration$==Length$-1)",&t);
  fMax.GetNdata();
  t.GetEntry(0);
  std::cout << " fMin = " << fMin.EvalInstance() << std::endl;
  std::cout << " fMax = " << fMax.EvalInstance() << std::endl;


  assert(fMin.EvalInstance()==4);
  assert(fMax.EvalInstance()==4);
  return 0;
}

