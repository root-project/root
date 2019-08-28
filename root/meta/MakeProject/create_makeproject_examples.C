R__LOAD_LIBRARY(stl_makeproject_test)

#include <stl_makeproject_test.h>

bool file_exists (const std::string &file_name) {
   return gSystem->AccessPathName(file_name.c_str(), kWritePermission);
}
int create_makeproject_examples()
{
   if (file_exists("./disabled.module.modulemap")) {
      gSystem->Rename( "./disabled.module.modulemap" , "./module.modulemap");
   }
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
   gSystem->Rename( "./module.modulemap" , "./disabled.module.modulemap");
   return 0;
}
