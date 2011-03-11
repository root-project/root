#include "TTree.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "mytypes.h"
#include <iostream>
#include <cassert>

#ifdef __MAKECINT__
#pragma link C++ class mytypes::Trajectory+;
#pragma link C++ class mytypes::MyEntry+;
#pragma link C++ class mytypes::Collection+;
#pragma link C++ class std::vector<mytypes::MyEntry>+;
#pragma link C++ class std::vector<int>+;
#endif


void check (const mytypes::MyEntry& e,
            unsigned int i, unsigned int j)
{
  for (unsigned int k=0; k < 8; k++) {
    assert (e.foo[k] == i*10+j*5+k);
  }

  assert (e.m_trajectory.size() == j+1);
  for (unsigned int k=0; k < j+1; k++) {
     assert (e.m_trajectory[k] == (int)(i*100+j*10+k));
  }
}



void test1 (TFile& f)
{
  std::cout << "test1\n";
  TTree* t = (TTree*)f.Get("tree");
  mytypes::Collection* c = new mytypes::Collection;
  TBranch* b = t->GetBranch ("T1");
  b->SetAddress (&c);
  Long64_t n = t->GetEntries();
  for (Long64_t i=0; i < n; i++) {
    b->GetEntry(i);
    assert (c->m_entries.size() == (unsigned int)i+1);
    for (unsigned int j=0; j < c->m_entries.size(); j++)
      check (c->m_entries[j], i, j);
  }
}




int assert_tread()
{
  TFile f ("tcls.root");
  if (f.IsZombie()) {
     return 1;
  }
  test1 (f);
  return 0;
}
