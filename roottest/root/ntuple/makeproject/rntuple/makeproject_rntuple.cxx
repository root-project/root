#include <TFile.h>

int main()
{
   std::unique_ptr<TFile> f{TFile::Open("ntuple_makeproject_stl_example_rntuple.root")};
   if (!f || f->IsZombie())
      return 1;
   f->MakeProject("librntuplestltest", "*", "recreate++");
   return 0;
}
