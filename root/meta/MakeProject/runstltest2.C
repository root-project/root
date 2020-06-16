R__LOAD_LIBRARY(stltest/stltest)

#include "stltest/SillyStlEvent.h"
#include "TTree.h"
#include "TFile.h"
#include "TSystem.h"

int test_event()
{
   TTree *T = (TTree*)gFile->Get("T");
   SillyStlEvent *event = new SillyStlEvent();
   event->foo = 0xfa3;
   TBranch *branch = T->GetBranch("test");
   branch->SetAddress(&event);
   T->GetEvent(0);
   return event->foo.to_ulong() != 0xfa2;
}

int runstltest2()
{
   TFile *_file0 = TFile::Open("stl_example.root");
   //int load_result = gSystem->Load("stltest/stltest.so");
   //if (load_result) {return load_result;}
   return test_event();
}

