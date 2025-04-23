#include "TTree.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "mytypes.h"


#ifdef __MAKECINT__
#pragma link C++ class mytypes::FirstBase+;
#pragma link C++ class mytypes::OtherBase+;
#pragma link C++ class mytypes::Trajectory+;
#pragma link C++ class mytypes::MyEntry+;
#pragma link C++ class mytypes::Collection+;
#pragma link C++ class std::vector<mytypes::MyEntry>+;
#pragma link C++ class std::vector<int>+;
#endif

mytypes::MyEntry make_entry (int i, int j)
{
  mytypes::MyEntry e;
  for (int k=0; k < 8; k++) {
    e.foo[k] = i*10+j*5+k;
  }

  e.m_trajectory.fA = 10000 + i;
  e.m_trajectory.fB = 10000 + j;
  e.m_trajectory.fPx = 1000 + i; 
  e.m_trajectory.fPy = 1000 + j;
  e.m_trajectory.resize(j+1);
  for (int k=0; k < j+1; k++) {
    e.m_trajectory[k] = i*100+j*10+k;
  }

  return e;
}

void test1(TFile& f)
{
  TTree tree ("tree", "tree");
  mytypes::Collection* c = new mytypes::Collection;
  tree.Branch ("T1", &c, 16000, 99);
  for (int i=0; i < 10; i++) {
     c->clear();
     for (int jo=0; jo < (2*i+1); jo++)
        c->push_back (make_entry (jo, i));
     c->m_entries.clear();
     for (int j=0; j < i+1; j++)
        c->m_entries.push_back (make_entry (i, j));
     tree.Fill();
  }
  tree.Print();
  f.Write();
}


int assert_twrite()
{
  TFile f ("tcls.root", "RECREATE");
  if (f.IsZombie()) {
     return 1;
  }
  test1 (f);
  return 0;
}
