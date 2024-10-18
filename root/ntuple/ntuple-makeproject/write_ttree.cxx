#include "ntuple_makeproject_header.h"
#include <TFile.h>
#include <TTree.h>

int main()
{
   TFile f("ntuple_makeproject_stl_example_ttree.root", "RECREATE");
   MySTLEvent event{};
   TTree t("events", "test events");
   t.Branch("test", "MySTLEvent", &event);
   t.Fill();
   f.Write();
   return 0;
}
