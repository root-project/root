// this test program compares the I/O performance obtained with
// all STL collections of objects or pointers to objects and also
// Root collection class TClonesArray.
// Trees in compression and non compression mode are created for each
// of the following cases:
//  -STLcollection<THit>
//  -STLcollection<THit*>
//  -TClonesArray(TObjHit) in no split mode
//  -TClonesArray(TObjHit) in split mode
// where:
//  THit is a class not derived from TObject
//  TObjHit derives from TObject and THit
//
//  run with
//     bench
//   or
//     bench -m   to stream objects memberwise
//
// The test prints a summary table comparing performances for all above cases
// (CPU, file size, compression factors).
// Reference numbers on a Pentium IV 2.4 Ghz machine are given as reference.
//      Authors:  Rene Brun, Markus Frank

#include "TROOT.h"
#include "TClonesArray.h"
#include "TStopwatch.h"
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TStreamerInfo.h"

#include "TBench.h"


int main(int argc, char **argv)
{
  // by default stream objects objectwise
  // if program option "-m" is specified, stream memberwise
  if (argc > 1) {
     if (strstr(argv[1],"-m")) TStreamerInfo::SetStreamMemberWise();
     printf("bench option -m specified. Streaming objects memberwise\n");
  }
  int nhits       = 1000;
  int nevents     = 400;
  Float_t cx;

  Double_t cptot = 0;
  TStopwatch timer;

  //delete temp file used for the benchmark
  gSystem->Exec("rm -f /tmp/bench.root");
  
  /// STL VECTOR
  timer.Start();
  TSTLhit *STLhit = new TSTLhit(nhits);
  STLhit->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt1 = timer.RealTime();
  Double_t cp1 = timer.CpuTime();
  cptot += cp1;
  printf("1 vector    : RT=%6.2f s  Cpu=%6.2f s\n",rt1,cp1);
  timer.Start(kTRUE);
  Int_t nbytes1 = STLhit->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt2w = timer.RealTime();
  Double_t cp2w = timer.CpuTime();
  cptot += cp2w;
  printf("2 vector   w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt2w-rt1,cp2w-cp1,nbytes1,cx);
  timer.Start(kTRUE);
  STLhit->ReadTree();
  timer.Stop();
  Double_t rt2r = timer.RealTime();
  Double_t cp2r = timer.CpuTime();
  cptot += cp2r;
  printf("3 vector   r: RT=%6.2f s  Cpu=%6.2f s\n",rt2r,cp2r);
  timer.Start(kTRUE);
  Float_t cx3;
  Int_t nbytes3 = STLhit->MakeTree(1,nevents,1,99,cx3);
  timer.Stop();
  Double_t rt3w = timer.RealTime();
  Double_t cp3w = timer.CpuTime();
  cptot += cp3w;
  printf("4 vector   w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt3w-rt1,cp3w-cp1,nbytes3,cx3);
  timer.Start(kTRUE);
  STLhit->ReadTree();
  timer.Stop();
  Double_t rt3r = timer.RealTime();
  Double_t cp3r = timer.CpuTime();
  cptot += cp3r;
  printf("5 vector   r: RT=%6.2f s  Cpu=%6.2f s\n",rt3r,cp3r);
  delete STLhit;

  // STL list
  timer.Start();
  TSTLhitList *STLhit_list = new TSTLhitList(nhits);
  STLhit_list->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt1L = timer.RealTime();
  Double_t cp1L = timer.CpuTime();
  cptot += cp1L;
  printf("1 list      : RT=%6.2f s  Cpu=%6.2f s\n",rt1L,cp1L);
  timer.Start(kTRUE);
  Int_t nbytes1L = STLhit_list->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt2wL = timer.RealTime();
  Double_t cp2wL = timer.CpuTime();
  cptot += cp2wL;
  printf("2 list     w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt2wL-rt1L,cp2wL-cp1L,nbytes1L,cx);
  timer.Start(kTRUE);
  STLhit_list->ReadTree();
  timer.Stop();
  Double_t rt2rL = timer.RealTime();
  Double_t cp2rL = timer.CpuTime();
  cptot += cp2rL;
  printf("3 list     r: RT=%6.2f s  Cpu=%6.2f s\n",rt2rL,cp2rL);
  timer.Start(kTRUE);
  Float_t cx3L;
  Int_t nbytes3L = STLhit_list->MakeTree(1,nevents,1,99,cx3L);
  timer.Stop();
  Double_t rt3wL = timer.RealTime();
  Double_t cp3wL = timer.CpuTime();
  cptot += cp3wL;
  printf("4 list     w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt3wL-rt1L,cp3wL-cp1L,nbytes3L,cx3L);
  timer.Start(kTRUE);
  STLhit_list->ReadTree();
  timer.Stop();
  Double_t rt3rL = timer.RealTime();
  Double_t cp3rL = timer.CpuTime();
  cptot += cp3rL;
  printf("5 list     r: RT=%6.2f s  Cpu=%6.2f s\n",rt3rL,cp3rL);
  delete STLhit_list;

  // STL DEQUE
  timer.Start();
  TSTLhitDeque *STLhit_deque = new TSTLhitDeque(nhits);
  STLhit_deque->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt1D = timer.RealTime();
  Double_t cp1D = timer.CpuTime();
  cptot += cp1D;
  printf("1 deque     : RT=%6.2f s  Cpu=%6.2f s\n",rt1D,cp1D);
  timer.Start(kTRUE);
  Int_t nbytes1D = STLhit_deque->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt2wD = timer.RealTime();
  Double_t cp2wD = timer.CpuTime();
  cptot += cp2wD;
  printf("2 deque    w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt2wD-rt1D,cp2wD-cp1D,nbytes1D,cx);
  timer.Start(kTRUE);
  STLhit_deque->ReadTree();
  timer.Stop();
  Double_t rt2rD = timer.RealTime();
  Double_t cp2rD = timer.CpuTime();
  cptot += cp2rD;
  printf("3 deque    r: RT=%6.2f s  Cpu=%6.2f s\n",rt2rD,cp2rD);
  timer.Start(kTRUE);
  Float_t cx3D;
  Int_t nbytes3D = STLhit_deque->MakeTree(1,nevents,1,99,cx3D);
  timer.Stop();
  Double_t rt3wD = timer.RealTime();
  Double_t cp3wD = timer.CpuTime();
  cptot += cp3wD;
  printf("4 deque    w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt3wD-rt1D,cp3wD-cp1D,nbytes3D,cx3D);
  timer.Start(kTRUE);
  STLhit_deque->ReadTree();
  timer.Stop();
  Double_t rt3rD = timer.RealTime();
  Double_t cp3rD = timer.CpuTime();
  cptot += cp3rD;
  printf("5 deque    r: RT=%6.2f s  Cpu=%6.2f s\n",rt3rD,cp3rD);
  delete STLhit_deque;

  // STL SET
  timer.Start();
  TSTLhitSet *STLhit_set = new TSTLhitSet(nhits);
  STLhit_set->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt1S = timer.RealTime();
  Double_t cp1S = timer.CpuTime();
  cptot += cp1S;
  printf("1 set       : RT=%6.2f s  Cpu=%6.2f s\n",rt1S,cp1S);
  timer.Start(kTRUE);
  Int_t nbytes1S = STLhit_set->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt2wS = timer.RealTime();
  Double_t cp2wS = timer.CpuTime();
  cptot += cp2wS;
  printf("2 set      w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt2wS-rt1S,cp2wS-cp1S,nbytes1S,cx);
  timer.Start(kTRUE);
  STLhit_set->ReadTree();
  timer.Stop();
  Double_t rt2rS = timer.RealTime();
  Double_t cp2rS = timer.CpuTime();
  cptot += cp2rS;
  printf("3 set      r: RT=%6.2f s  Cpu=%6.2f s\n",rt2rS,cp2rS);
  timer.Start(kTRUE);
  Float_t cx3S;
  Int_t nbytes3S = STLhit_set->MakeTree(1,nevents,1,99,cx3S);
  timer.Stop();
  Double_t rt3wS = timer.RealTime();
  Double_t cp3wS = timer.CpuTime();
  cptot += cp3wS;
  printf("4 set      w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt3wS-rt1S,cp3wS-cp1S,nbytes3S,cx3S);
  timer.Start(kTRUE);
  STLhit_set->ReadTree();
  timer.Stop();
  Double_t rt3rS = timer.RealTime();
  Double_t cp3rS = timer.CpuTime();
  cptot += cp3rS;
  printf("5 set      r: RT=%6.2f s  Cpu=%6.2f s\n",rt3rS,cp3rS);
  delete STLhit_set;

  // STL MULTI SET
  timer.Start();
  TSTLhitMultiset *STLhit_multiset = new TSTLhitMultiset(nhits);
  STLhit_multiset->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt1M = timer.RealTime();
  Double_t cp1M = timer.CpuTime();
  cptot += cp1M;
  printf("1 multiset  : RT=%6.2f s  Cpu=%6.2f s\n",rt1M,cp1M);
  timer.Start(kTRUE);
  Int_t nbytes1M = STLhit_multiset->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt2wM = timer.RealTime();
  Double_t cp2wM = timer.CpuTime();
  cptot += cp2wM;
  printf("2 multiset w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt2wM-rt1M,cp2wM-cp1M,nbytes1M,cx);
  timer.Start(kTRUE);
  STLhit_multiset->ReadTree();
  timer.Stop();
  Double_t rt2rM = timer.RealTime();
  Double_t cp2rM = timer.CpuTime();
  cptot += cp2rM;
  printf("3 multiset r: RT=%6.2f s  Cpu=%6.2f s\n",rt2rM,cp2rM);
  timer.Start(kTRUE);
  Float_t cx3M;
  Int_t nbytes3M = STLhit_multiset->MakeTree(1,nevents,1,99,cx3M);
  timer.Stop();
  Double_t rt3wM = timer.RealTime();
  Double_t cp3wM = timer.CpuTime();
  cptot += cp3wM;
  printf("4 multiset w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt3wM-rt1M,cp3wM-cp1M,nbytes3M,cx3M);
  timer.Start(kTRUE);
  STLhit_multiset->ReadTree();
  timer.Stop();
  Double_t rt3rM = timer.RealTime();
  Double_t cp3rM = timer.CpuTime();
  cptot += cp3rM;
  printf("5 multiset r: RT=%6.2f s  Cpu=%6.2f s\n",rt3rM,cp3rM);
  delete STLhit_multiset;

  // STL map
  timer.Start();
  TSTLhitMap *STLhit_map = new TSTLhitMap(nhits);
  STLhit_map->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt1MAP = timer.RealTime();
  Double_t cp1MAP = timer.CpuTime();
  cptot += cp1MAP;
  printf("1 map       : RT=%6.2f s  Cpu=%6.2f s\n",rt1MAP,cp1MAP);
  timer.Start(kTRUE);
  Int_t nbytes1MAP = STLhit_map->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt2wMAP = timer.RealTime();
  Double_t cp2wMAP = timer.CpuTime();
  cptot += cp2wMAP;
  printf("2 map      w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt2wMAP-rt1MAP,cp2wMAP-cp1MAP,nbytes1MAP,cx);
  timer.Start(kTRUE);
  STLhit_map->ReadTree();
  timer.Stop();
  Double_t rt2rMAP = timer.RealTime();
  Double_t cp2rMAP = timer.CpuTime();
  cptot += cp2rMAP;
  printf("3 map      r: RT=%6.2f s  Cpu=%6.2f s\n",rt2rMAP,cp2rMAP);
  timer.Start(kTRUE);
  Float_t cx3MAP;
  Int_t nbytes3MAP = STLhit_map->MakeTree(1,nevents,1,99,cx3MAP);
  timer.Stop();
  Double_t rt3wMAP = timer.RealTime();
  Double_t cp3wMAP = timer.CpuTime();
  cptot += cp3wMAP;
  printf("4 map      w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt3wMAP-rt1MAP,cp3wMAP-cp1MAP,nbytes3MAP,cx3MAP);
  timer.Start(kTRUE);
  STLhit_map->ReadTree();
  timer.Stop();
  Double_t rt3rMAP = timer.RealTime();
  Double_t cp3rMAP = timer.CpuTime();
  cptot += cp3rMAP;
  printf("5 map      r: RT=%6.2f s  Cpu=%6.2f s\n",rt3rMAP,cp3rMAP);
  delete STLhit_map;

  // STL multimap
  timer.Start();
  TSTLhitMultiMap *STLhit_multimap = new TSTLhitMultiMap(nhits);
  STLhit_multimap->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt1MMAP = timer.RealTime();
  Double_t cp1MMAP = timer.CpuTime();
  cptot += cp1MMAP;
  printf("1 multimap  : RT=%6.2f s  Cpu=%6.2f s\n",rt1MMAP,cp1MMAP);
  timer.Start(kTRUE);
  Int_t nbytes1MMAP = STLhit_multimap->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt2wMMAP = timer.RealTime();
  Double_t cp2wMMAP = timer.CpuTime();
  cptot += cp2wMMAP;
  printf("2 multimap w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt2wMMAP-rt1MMAP,cp2wMMAP-cp1MMAP,nbytes1MMAP,cx);
  timer.Start(kTRUE);
  STLhit_multimap->ReadTree();
  timer.Stop();
  Double_t rt2rMMAP = timer.RealTime();
  Double_t cp2rMMAP = timer.CpuTime();
  cptot += cp2rMMAP;
  printf("3 multimap r: RT=%6.2f s  Cpu=%6.2f s\n",rt2rMMAP,cp2rMMAP);
  timer.Start(kTRUE);
  Float_t cx3MMAP;
  Int_t nbytes3MMAP = STLhit_multimap->MakeTree(1,nevents,1,99,cx3MMAP);
  timer.Stop();
  Double_t rt3wMMAP = timer.RealTime();
  Double_t cp3wMMAP = timer.CpuTime();
  cptot += cp3wMMAP;
  printf("4 multimap w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt3wMMAP-rt1MMAP,cp3wMMAP-cp1MMAP,nbytes3MMAP,cx3MMAP);
  timer.Start(kTRUE);
  STLhit_multimap->ReadTree();
  timer.Stop();
  Double_t rt3rMMAP = timer.RealTime();
  Double_t cp3rMMAP = timer.CpuTime();
  cptot += cp3rMMAP;
  printf("5 multimap r: RT=%6.2f s  Cpu=%6.2f s\n",rt3rMMAP,cp3rMMAP);
  delete STLhit_multimap;

  //testing STL vector of pointers to THit
  timer.Start();
  TSTLhitStar *STLhitStar = new TSTLhitStar(nhits);
  STLhitStar->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt4 = timer.RealTime();
  Double_t cp4 = timer.CpuTime();
  cptot += cp4;
  printf("1 vector*   : RT=%6.2f s  Cpu=%6.2f s\n",rt4,cp4);
  timer.Start(kTRUE);
  Int_t nbytes5 = STLhitStar->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt5w = timer.RealTime();
  Double_t cp5w = timer.CpuTime();
  cptot += cp5w;
  printf("2 vector*  w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt5w-rt4,cp5w-cp4,nbytes5,cx);
  timer.Start(kTRUE);
  STLhitStar->ReadTree();
  timer.Stop();
  Double_t rt5r = timer.RealTime();
  Double_t cp5r = timer.CpuTime();
  cptot += cp5r;
  printf("3 vector*  r: RT=%6.2f s  Cpu=%6.2f s\n",rt5r,cp5r);
  timer.Start(kTRUE);
  Float_t cx6;
  Int_t nbytes6 = STLhitStar->MakeTree(1,nevents,1,99,cx6);
  timer.Stop();
  Double_t rt6w = timer.RealTime();
  Double_t cp6w = timer.CpuTime();
  cptot += cp6w;
  printf("4 vector*  w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt6w-rt4,cp6w-cp4,nbytes6,cx6);
  timer.Start(kTRUE);
  STLhitStar->ReadTree();
  timer.Stop();
  Double_t rt6r = timer.RealTime();
  Double_t cp6r = timer.CpuTime();
  cptot += cp6r;
  printf("5 vector*  r: RT=%6.2f s  Cpu=%6.2f s\n",rt6r,cp6r);
  delete STLhitStar;

  // STL list*
  timer.Start();
  TSTLhitStarList *STLhit_liststar = new TSTLhitStarList(nhits);
  STLhit_liststar->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt1LS = timer.RealTime();
  Double_t cp1LS = timer.CpuTime();
  cptot += cp1LS;
  printf("1 list*     : RT=%6.2f s  Cpu=%6.2f s\n",rt1LS,cp1LS);
  timer.Start(kTRUE);
  Int_t nbytes1LS = STLhit_liststar->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt2wLS = timer.RealTime();
  Double_t cp2wLS = timer.CpuTime();
  cptot += cp2wLS;
  printf("2 list*    w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt2wLS-rt1LS,cp2wLS-cp1LS,nbytes1LS,cx);
  timer.Start(kTRUE);
  STLhit_liststar->ReadTree();
  timer.Stop();
  Double_t rt2rLS = timer.RealTime();
  Double_t cp2rLS = timer.CpuTime();
  cptot += cp2rLS;
  printf("3 list*    r: RT=%6.2f s  Cpu=%6.2f s\n",rt2rLS,cp2rLS);
  timer.Start(kTRUE);
  Float_t cx3LS;
  Int_t nbytes3LS = STLhit_liststar->MakeTree(1,nevents,1,99,cx3LS);
  timer.Stop();
  Double_t rt3wLS = timer.RealTime();
  Double_t cp3wLS = timer.CpuTime();
  cptot += cp3wLS;
  printf("4 list*    w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt3wLS-rt1LS,cp3wLS-cp1LS,nbytes3LS,cx3LS);
  timer.Start(kTRUE);
  STLhit_liststar->ReadTree();
  timer.Stop();
  Double_t rt3rLS = timer.RealTime();
  Double_t cp3rLS = timer.CpuTime();
  cptot += cp3rLS;
  printf("5 list*    r: RT=%6.2f s  Cpu=%6.2f s\n",rt3rLS,cp3rLS);
  delete STLhit_liststar;

  // STL DEQUE*
  timer.Start();
  TSTLhitStarDeque *STLhit_dequestar = new TSTLhitStarDeque(nhits);
  STLhit_dequestar->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt1DS = timer.RealTime();
  Double_t cp1DS = timer.CpuTime();
  cptot += cp1DS;
  printf("1 deque*    : RT=%6.2f s  Cpu=%6.2f s\n",rt1DS,cp1DS);
  timer.Start(kTRUE);
  Int_t nbytes1DS = STLhit_dequestar->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt2wDS = timer.RealTime();
  Double_t cp2wDS = timer.CpuTime();
  cptot += cp2wDS;
  printf("2 deque*   w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt2wDS-rt1DS,cp2wDS-cp1DS,nbytes1DS,cx);
  timer.Start(kTRUE);
  STLhit_dequestar->ReadTree();
  timer.Stop();
  Double_t rt2rDS = timer.RealTime();
  Double_t cp2rDS = timer.CpuTime();
  cptot += cp2rDS;
  printf("3 deque*   r: RT=%6.2f s  Cpu=%6.2f s\n",rt2rDS,cp2rDS);
  timer.Start(kTRUE);
  Float_t cx3DS;
  Int_t nbytes3DS = STLhit_dequestar->MakeTree(1,nevents,1,99,cx3DS);
  timer.Stop();
  Double_t rt3wDS = timer.RealTime();
  Double_t cp3wDS = timer.CpuTime();
  cptot += cp3wDS;
  printf("4 deque*   w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt3wDS-rt1DS,cp3wDS-cp1DS,nbytes3DS,cx3DS);
  timer.Start(kTRUE);
  STLhit_dequestar->ReadTree();
  timer.Stop();
  Double_t rt3rDS = timer.RealTime();
  Double_t cp3rDS = timer.CpuTime();
  cptot += cp3rDS;
  printf("5 deque*   r: RT=%6.2f s  Cpu=%6.2f s\n",rt3rDS,cp3rDS);
  delete STLhit_dequestar;

  // STL SET*
  timer.Start();
  TSTLhitStarSet *STLhit_setstar = new TSTLhitStarSet(nhits);
  STLhit_setstar->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt1SS = timer.RealTime();
  Double_t cp1SS = timer.CpuTime();
  cptot += cp1SS;
  printf("1 set*      : RT=%6.2f s  Cpu=%6.2f s\n",rt1SS,cp1SS);
  timer.Start(kTRUE);
  Int_t nbytes1SS = STLhit_setstar->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt2wSS = timer.RealTime();
  Double_t cp2wSS = timer.CpuTime();
  cptot += cp2wSS;
  printf("2 set*     w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt2wSS-rt1SS,cp2wSS-cp1SS,nbytes1SS,cx);
  timer.Start(kTRUE);
  STLhit_setstar->ReadTree();
  timer.Stop();
  Double_t rt2rSS = timer.RealTime();
  Double_t cp2rSS = timer.CpuTime();
  cptot += cp2rSS;
  printf("3 set*     r: RT=%6.2f s  Cpu=%6.2f s\n",rt2rSS,cp2rSS);
  timer.Start(kTRUE);
  Float_t cx3SS;
  Int_t nbytes3SS = STLhit_setstar->MakeTree(1,nevents,1,99,cx3SS);
  timer.Stop();
  Double_t cp3wSS = timer.CpuTime();
  cptot += cp3wSS;
  printf("4 set*     w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt3wDS-rt1DS,cp3wSS-cp1SS,nbytes3SS,cx3SS);
  timer.Start(kTRUE);
  STLhit_setstar->ReadTree();
  timer.Stop();
  Double_t rt3rSS = timer.RealTime();
  Double_t cp3rSS = timer.CpuTime();
  cptot += cp3rSS;
  printf("5 set*      : RT=%6.2f s  Cpu=%6.2f s\n",rt3rSS,cp3rSS);
  delete STLhit_setstar;

  // STL MULTI SET*
  timer.Start();
  TSTLhitStarMultiSet *STLhit_multisetstar = new TSTLhitStarMultiSet(nhits);
  STLhit_multisetstar->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt1MS = timer.RealTime();
  Double_t cp1MS = timer.CpuTime();
  cptot += cp1MS;
  printf("1 multiset* : RT=%6.2f s  Cpu=%6.2f s\n",rt1MS,cp1MS);
  timer.Start(kTRUE);
  Int_t nbytes1MS = STLhit_multisetstar->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt2wMS = timer.RealTime();
  Double_t cp2wMS = timer.CpuTime();
  cptot += cp2wMS;
  printf("2 multiset*w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt2wMS-rt1MS,cp2wMS-cp1MS,nbytes1MS,cx);
  timer.Start(kTRUE);
  STLhit_multisetstar->ReadTree();
  timer.Stop();
  Double_t rt2rMS = timer.RealTime();
  Double_t cp2rMS = timer.CpuTime();
  cptot += cp2rMS;
  printf("3 multiset*r: RT=%6.2f s  Cpu=%6.2f s\n",rt2rMS,cp2rMS);
  timer.Start(kTRUE);
  Float_t cx3MS;
  Int_t nbytes3MS = STLhit_multisetstar->MakeTree(1,nevents,1,99,cx3MS);
  timer.Stop();
  Double_t cp3wMS = timer.CpuTime();
  cptot += cp3wMS;
  printf("4 multiset*w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt3wDS-rt1DS,cp3wDS-cp1DS,nbytes3DS,cx3DS);
  timer.Start(kTRUE);
  STLhit_multisetstar->ReadTree();
  timer.Stop();
  Double_t rt3rMS = timer.RealTime();
  Double_t cp3rMS = timer.CpuTime();
  cptot += cp3rMS;
  printf("5 multiset* : RT=%6.2f s  Cpu=%6.2f s\n",rt3rMS,cp3rMS);
  delete STLhit_multisetstar;

  // STL MAP*
  timer.Start();
  TSTLhitStarMap *STLhit_mapstar = new TSTLhitStarMap(nhits);
  STLhit_mapstar->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt1MAPS = timer.RealTime();
  Double_t cp1MAPS = timer.CpuTime();
  cptot += cp1MAPS;
  printf("1 map*      : RT=%6.2f s  Cpu=%6.2f s\n",rt1MAPS,cp1MAPS);
  timer.Start(kTRUE);
  Int_t nbytes1MAPS = STLhit_mapstar->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt2wMAPS = timer.RealTime();
  Double_t cp2wMAPS = timer.CpuTime();
  cptot += cp2wMAPS;
  printf("2 map*     w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt2wMAPS-rt1MAPS,cp2wMAPS-cp1MAPS,nbytes1MAPS,cx);
  timer.Start(kTRUE);
  STLhit_mapstar->ReadTree();
  timer.Stop();
  Double_t rt2rMAPS = timer.RealTime();
  Double_t cp2rMAPS = timer.CpuTime();
  cptot += cp2rMAPS;
  printf("3 map*     r: RT=%6.2f s  Cpu=%6.2f s\n",rt2rMAPS,cp2rMAPS);
  timer.Start(kTRUE);
  Float_t cx3MAPS;
  Int_t nbytes3MAPS = STLhit_mapstar->MakeTree(1,nevents,1,99,cx3MAPS);
  timer.Stop();
  Double_t rt3wMAPS = timer.RealTime();
  Double_t cp3wMAPS = timer.CpuTime();
  cptot += cp3wMAPS;
  printf("4 map*     w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt3wMAPS-rt1MAPS,cp3wMAPS-cp1MAPS,nbytes3MAPS,cx3MAPS);
  timer.Start(kTRUE);
  STLhit_mapstar->ReadTree();
  timer.Stop();
  Double_t rt3rMAPS = timer.RealTime();
  Double_t cp3rMAPS = timer.CpuTime();
  cptot += cp3rMAPS;
  printf("5 map*      : RT=%6.2f s  Cpu=%6.2f s\n",rt3rMAPS,cp3rMAPS);
  delete STLhit_mapstar;

  // STL MULTIMAP*
  timer.Start();
  TSTLhitStarMultiMap *STLhit_multimapstar = new TSTLhitStarMultiMap(nhits);
  STLhit_multimapstar->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt1MMAPS = timer.RealTime();
  Double_t cp1MMAPS = timer.CpuTime();
  cptot += cp1MMAPS;
  printf("1 multimap* : RT=%6.2f s  Cpu=%6.2f s\n",rt1MMAPS,cp1MMAPS);
  timer.Start(kTRUE);
  Int_t nbytes1MMAPS = STLhit_multimapstar->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt2wMMAPS = timer.RealTime();
  Double_t cp2wMMAPS = timer.CpuTime();
  cptot += cp2wMMAPS;
  printf("2 multimap*w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt2wMMAPS-rt1MMAPS,cp2wMMAPS-cp1MMAPS,nbytes1MMAPS,cx);
  timer.Start(kTRUE);
  STLhit_multimapstar->ReadTree();
  timer.Stop();
  Double_t rt2rMMAPS = timer.RealTime();
  Double_t cp2rMMAPS = timer.CpuTime();
  cptot += cp2rMMAPS;
  printf("3 multimap*r: RT=%6.2f s  Cpu=%6.2f s\n",rt2rMMAPS,cp2rMMAPS);
  timer.Start(kTRUE);
  Float_t cx3MMAPS;
  Int_t nbytes3MMAPS = STLhit_multimapstar->MakeTree(1,nevents,1,99,cx3MMAPS);
  timer.Stop();
  Double_t cp3wMMAPS = timer.CpuTime();
  cptot += cp3wMMAPS;
  printf("4 multimap*w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt3wMAPS-rt1MAPS,cp3wMAPS-cp1MAPS,nbytes3MAPS,cx3MAPS);
  timer.Start(kTRUE);
  STLhit_multimapstar->ReadTree();
  timer.Stop();
  Double_t rt3rMMAPS = timer.RealTime();
  Double_t cp3rMMAPS = timer.CpuTime();
  cptot += cp3rMMAPS;
  printf("5 multimap* : RT=%6.2f s  Cpu=%6.2f s\n",rt3rMMAPS,cp3rMMAPS);
  delete STLhit_multimapstar;
  
  //testing TClonesArray of TObjHit deriving from THit
  timer.Start();
  TCloneshit *Cloneshit = new TCloneshit(nhits);
  Cloneshit->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt7 = timer.RealTime();
  Double_t cp7 = timer.CpuTime();
  cptot += cp7;
  printf("1 Clones1   : RT=%6.2f s  Cpu=%6.2f s\n",rt7,cp7);
  timer.Start(kTRUE);
  Int_t nbytes8 = Cloneshit->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt8w = timer.RealTime();
  Double_t cp8w = timer.CpuTime();
  cptot += cp8w;
  printf("2 Clones1  w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt8w-rt7,cp8w-cp7,nbytes8,cx);
  timer.Start(kTRUE);
  Cloneshit->ReadTree();
  timer.Stop();
  Double_t rt8r = timer.RealTime();
  Double_t cp8r = timer.CpuTime();
  cptot += cp8r;
  printf("3 Clones1  r: RT=%6.2f s  Cpu=%6.2f s\n",rt8r,cp8r);
  timer.Start(kTRUE);
  Float_t cx9;
  Int_t nbytes9 = Cloneshit->MakeTree(1,nevents,1,99,cx9);
  timer.Stop();
  Double_t rt9w = timer.RealTime();
  Double_t cp9w = timer.CpuTime();
  cptot += cp9w;
  printf("4 Clones1  w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt9w-rt7,cp9w-cp7,nbytes9,cx9);
  timer.Start(kTRUE);
  Cloneshit->ReadTree();
  timer.Stop();
  Double_t rt9r = timer.RealTime();
  Double_t cp9r = timer.CpuTime();
  cptot += cp9r;
  printf("5 Clones1  r: RT=%6.2f s  Cpu=%6.2f s\n",rt9r,cp9r);
  timer.Start(kTRUE);
  Int_t nbytes10 = Cloneshit->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt10w = timer.RealTime();
  Double_t cp10w = timer.CpuTime();
  cptot += cp10w;
  printf("6 Clones2  w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt10w-rt7,cp10w-cp7,nbytes10,cx);
  timer.Start(kTRUE);
  Cloneshit->ReadTree();
  timer.Stop();
  Double_t rt10r = timer.RealTime();
  Double_t cp10r = timer.CpuTime();
  cptot += cp10r;
  printf("7 Clones2  r: RT=%6.2f s  Cpu=%6.2f s\n",rt10r,cp10r);
  timer.Start(kTRUE);
  Float_t cx11;
  Int_t nbytes11 = Cloneshit->MakeTree(1,nevents,1,99,cx11);
  timer.Stop();
  Double_t rt11w = timer.RealTime();
  Double_t cp11w = timer.CpuTime();
  cptot += cp11w;
  printf("8 Clones2  w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt11w-rt7,cp11w-cp7,nbytes11,cx11);
  timer.Start(kTRUE);
  Cloneshit->ReadTree();
  timer.Stop();
  Double_t rt11r = timer.RealTime();
  Double_t cp11r = timer.CpuTime();
  cptot += cp11r;
  printf("9 Clones2  r: RT=%6.2f s  Cpu=%6.2f s\n",rt11r,cp11r);
  Double_t cpref = 152.93;
  Double_t rootmarks = cpref*600/cptot;

  //print all results
  char line1[100], line2[100];
  printf("\n");
  printf("******************************************************************************\n");
  sprintf(line1,"Comparing STL vector with TClonesArray: Root %-8s",gROOT->GetVersion());
  printf("*       %s                *\n",line1);
  Bool_t UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
  if (UNIX) {
     FILE *fp = gSystem->OpenPipe("uname -a", "r");
     char line[60];
     fgets(line,60,fp); line[59] = 0;
     sprintf(line2,"%s",line);
     printf("*  %s\n",line);
     gSystem->ClosePipe(fp);
  } else {
     const char *os = gSystem->Getenv("OS");
     sprintf(line2,"Windows");
     if (!os) printf("*  Windows 95\n");
     else     printf("*  %s %s \n",os,gSystem->Getenv("PROCESSOR_IDENTIFIER"));
  }
  printf("*     Reference machine pcbrun.cern.ch  RedHat Linux 7.3                     *\n");
  printf("*         (Pentium IV 2.4 Ghz 512 Mbytes RAM, IDE disk)                      *\n");
  printf("*           (send your results to rootdev@root.cern.ch)                      *\n");
  printf("******************************************************************************\n");
  printf("* Time to fill the structures (seconds)   Reference      cx      Reference   *\n");
  printf("******************************************************************************\n");
  printf("* vector<THit>                  %6.2f        0.98     %5.2f        5.38     *\n",cp1,cx3);
  printf("* list<THit>                    %6.2f        1.10     %5.2f        5.38     *\n",cp1L,cx3L);
  printf("* deque<THit>                   %6.2f        1.01     %5.2f        5.38     *\n",cp1D,cx3D);
  printf("* set<THit>                     %6.2f        1.19     %5.2f        6.22     *\n",cp1S,cx3S);
  printf("* multiset<THit>                %6.2f        1.16     %5.2f        6.22     *\n",cp1M,cx3M);
  printf("* map<int,THit>                 %6.2f        1.95     %5.2f        5.26     *\n",cp1MAP,cx3MAP);
  printf("* multimap<int,THit>            %6.2f        1.42     %5.2f        5.26     *\n",cp1MMAP,cx3MMAP);
  printf("* vector<THit*>                 %6.2f        0.91     %5.2f        3.73     *\n",cp4,cx6);
  printf("* list<THit*>                   %6.2f        1.06     %5.2f        3.73     *\n",cp1LS,cx3LS);
  printf("* deque<THit*>                  %6.2f        0.92     %5.2f        3.73     *\n",cp1DS,cx3DS);
  printf("* set<THit*>                    %6.2f        1.22     %5.2f        3.68     *\n",cp1SS,cx3SS);
  printf("* multiset<THit*>               %6.2f        1.19     %5.2f        3.68     *\n",cp1MS,cx3MS);
  printf("* map<int,THit*>                %6.2f        1.17     %5.2f        3.78     *\n",cp1MAPS,cx3MAPS);
  printf("* multimap<int,THit*>           %6.2f        1.16     %5.2f        3.78     *\n",cp1MMAPS,cx3MMAPS);
  printf("* TClonesArray(TObjHit)         %6.2f        0.76     %5.2f        5.38     *\n",cp7,cx9);
  printf("* TClonesArray(TObjHit) split   %6.2f        0.76     %5.2f        5.38     *\n",cp7,cx11);
  printf("******************************************************************************\n");
  printf("* Size of file in bytes         comp 0    Reference    comp 1    Reference   *\n");
  printf("******************************************************************************\n");
  printf("* vector<THit>                  %8d   39725057   %8d   7386553     *\n",nbytes1,nbytes3);
  printf("* list<THit>                    %8d   39725099   %8d   7386439     *\n",nbytes1L,nbytes3L);
  printf("* deque<THit>                   %8d   39725106   %8d   7386333     *\n",nbytes1D,nbytes3D);
  printf("* set<THit>                     %8d   39725093   %8d   6389527     *\n",nbytes1S,nbytes3S);
  printf("* multiset<THit>                %8d   39725135   %8d   6389494     *\n",nbytes1M,nbytes3M);
  printf("* map<int,THit>                 %8d   41336713   %8d   7853614     *\n",nbytes1MAP,nbytes3MAP);
  printf("* multimap<int,THit>            %8d   41336755   %8d   7854089     *\n",nbytes1MMAP,nbytes3MMAP);
  printf("* vector<THit*>                 %8d   45257302   %8d  12141849     *\n",nbytes5,nbytes6);
  printf("* list<THit*>                   %8d   45257328   %8d  12140234     *\n",nbytes1LS,nbytes3LS);
  printf("* deque<THit*>                  %8d   45257341   %8d  12141094     *\n",nbytes1DS,nbytes3DS);
  printf("* set<THit*>                    %8d   45257324   %8d  12295977     *\n",nbytes1SS,nbytes3SS);
  printf("* multiset<THit*>               %8d   45257368   %8d  12293041     *\n",nbytes1MS,nbytes3MS);
  printf("* map<int,THit*>                %8d   46857329   %8d  12403158     *\n",nbytes1MAPS,nbytes3MAPS);
  printf("* multimap<int,THit*>           %8d   46857375   %8d  12403982     *\n",nbytes1MMAPS,nbytes3MMAPS);
  printf("* TClonesArray(TObjHit)         %8d   39723595   %8d   7385550     *\n",nbytes8,nbytes9);
  printf("* TClonesArray(TObjHit) split   %8d   39723595   %8d   7385295     *\n",nbytes10,nbytes11);
  printf("******************************************************************************\n");
  printf("* Time to write in seconds      comp 0    Reference    comp 1    Reference   *\n");
  printf("******************************************************************************\n");
  printf("* vector<THit>                  %6.2f        0.65    %6.2f        1.42     *\n",cp2w-cp1, cp3w-cp1);
  printf("* list<THit>                    %6.2f        0.47    %6.2f        1.58     *\n",cp2wL-cp1L,cp3wL-cp1L);
  printf("* deque<THit>                   %6.2f        0.45    %6.2f        1.53     *\n",cp2wD-cp1D,cp3wD-cp1D);
  printf("* set<THit>                     %6.2f        0.46    %6.2f        1.41     *\n",cp2wS-cp1S,cp3wS-cp1S);
  printf("* multiset<THit>                %6.2f        0.46    %6.2f        1.42     *\n",cp2wM-cp1M,cp3wM-cp1M);
  printf("* map<int,THit>                 %6.2f        0.51    %6.2f        1.74     *\n",cp2wMAP-cp1MAP,cp3wMAP-cp1MAP);
  printf("* multimap<int,THit>            %6.2f        0.51    %6.2f        1.75     *\n",cp2wMMAP-cp1MMAP,cp3wMMAP-cp1MMAP);
  printf("* vector<THit*>                 %6.2f        1.31    %6.2f        3.16     *\n",cp5w-cp1, cp6w-cp1);
  printf("* list<THit*>                   %6.2f        1.45    %6.2f        3.27     *\n",cp2wLS-cp1LS,cp3wLS-cp1LS);
  printf("* deque<THit*>                  %6.2f        1.37    %6.2f        3.22     *\n",cp2wDS-cp1DS,cp3wDS-cp1DS);
  printf("* set<THit*>                    %6.2f        1.48    %6.2f        3.32     *\n",cp2wSS-cp1SS,cp3wSS-cp1SS);
  printf("* multiset<THit*>               %6.2f        1.52    %6.2f        3.33     *\n",cp2wMS-cp1MS,cp3wMS-cp1MS);
  printf("* map<int,THit*>                %6.2f        1.55    %6.2f        3.45     *\n",cp2wMAPS-cp1MAPS,cp3wMAPS-cp1MAPS);
  printf("* multimap<int,THit*>           %6.2f        1.54    %6.2f        3.39     *\n",cp2wMMAPS-cp1MMAPS,cp3wMMAPS-cp1MMAPS);
  printf("* TClonesArray(TObjHit)         %6.2f        0.10    %6.2f        1.27     *\n",cp8w-cp1, cp9w-cp1);
  printf("* TClonesArray(TObjHit) split   %6.2f        0.12    %6.2f        1.25     *\n",cp10w-cp1,cp11w-cp1);
  printf("******************************************************************************\n");
  printf("* Time to read in seconds       comp 0    Reference    comp 1    Reference   *\n");
  printf("******************************************************************************\n");
  printf("* vector<THit>                  %6.2f        0.43    %6.2f        0.84     *\n",cp2r,cp3r);
  printf("* list<THit>                    %6.2f        0.67    %6.2f        1.10     *\n",cp2rL,cp3rL);
  printf("* deque<THit>                   %6.2f        0.53    %6.2f        0.96     *\n",cp2rD,cp3rD);
  printf("* set<THit>                     %6.2f        0.95    %6.2f        1.24     *\n",cp2rS,cp3rS);
  printf("* multiset<THit>                %6.2f        0.94    %6.2f        1.24     *\n",cp2rM,cp3rM);
  printf("* map<int,THit>                 %6.2f        0.98    %6.2f        1.43     *\n",cp2rMAP,cp3rMAP);
  printf("* multimap<int,THit>            %6.2f        0.98    %6.2f        1.43     *\n",cp2rMMAP,cp3rMMAP);
  printf("* vector<THit*>                 %6.2f        1.81    %6.2f        2.44     *\n",cp5r,cp6r);
  printf("* list<THit*>                   %6.2f        2.01    %6.2f        2.72     *\n",cp2rLS,cp3rLS);
  printf("* deque<THit*>                  %6.2f        1.86    %6.2f        2.46     *\n",cp2rDS,cp3rDS);
  printf("* set<THit*>                    %6.2f        2.13    %6.2f        2.82     *\n",cp2rSS,cp3rSS);
  printf("* multiset<THit*>               %6.2f        2.11    %6.2f        2.76     *\n",cp2rMS,cp3rMS);
  printf("* map<int,THit*>                %6.2f        2.14    %6.2f        2.81     *\n",cp2rMAPS,cp3rMAPS);
  printf("* multimap<int,THit*>           %6.2f        2.10    %6.2f        2.90     *\n",cp2rMMAPS,cp3rMMAPS);
  printf("* TClonesArray(TObjHit)         %6.2f        0.38    %6.2f        0.80     *\n",cp8r,cp9r);
  printf("* TClonesArray(TObjHit) split   %6.2f        0.36    %6.2f        0.79     *\n",cp10r,cp11r);
  printf("******************************************************************************\n");
  printf("* Total CPU time              %8.2f    %8.2f                           *\n",cptot,cpref);
  printf("* Estimated ROOTMARKS         %8.2f      600.00                           *\n",rootmarks);
  printf("******************************************************************************\n");
   return 0;
}
