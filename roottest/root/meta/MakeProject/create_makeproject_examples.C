R__LOAD_LIBRARY(stl_makeproject_test)

#include <stl_makeproject_test.h>

int create_makeproject_examples()
{
   gSystem->Unlink("./stl_example.root");
   TFile _file0("stl_example.root", "RECREATE");
   SillyStlEvent *event = nullptr;
   TTree tree("T", "test tree");
   tree.Branch("test", "SillyStlEvent", &event);
   event = new SillyStlEvent();
   tree.Fill();
   delete event;
   tree.Write();
   _file0.Close();
   gSystem->Unlink("./stl_makeproject_test.rootmap");
   return 0;
}
