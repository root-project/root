{
   TFile *_file0 = TFile::Open("stl_example.root");
   gFile->MakeProject("stltest","*","RECREATE++");
   //TTree *T = (TTree*)gFile->Get("T");
   //#include "stltest/SillyStlEvent.h"
   //SillyStlEvent *event = new SillyStlEvent();
   //return event->foo.to_ulong() == 0xfa2;
}
