#include <ROOT/RDataFrame.hxx>
#include <TFile.h>
#include <TROOT.h>
#include <TSystem.h>

using namespace ROOT;

void writeHistNoInputFile()
{
   std::cout << "write hist no input file" << std::endl;
   RDataFrame d(1);
   auto counter = 0;
   auto dd = d.Define("x", [&counter]() { return counter++; });
   auto c = dd.Count();

   TFile f("gdirectoryRestore1.root", "RECREATE");
   *c;
   assert(gDirectory == &f);
}

void makeDataSet()
{
   std::cout << "make dataset" << std::endl;
   RDataFrame d(1);
   auto counter = 0;
   auto dd = d.Define("x", [&counter]() { return counter++; });
   dd.Snapshot("t", "gdirectoryRestore2.root", {"x"});
}

void writeHistWithInputFile()
{
   std::cout << "write hist with input file" << std::endl;
   RDataFrame d("t", "gdirectoryRestore2.root");
   auto m = d.Histo1D<int>("x");
   TFile f("gdirectoryRestore3.root", "RECREATE");
   *m;
   assert(gDirectory == &f);
}

int main()
{
   ROOT::EnableImplicitMT();
   writeHistNoInputFile();
   makeDataSet();
   writeHistWithInputFile();

   gSystem->Unlink("gdirectoryRestore1.root");
   gSystem->Unlink("gdirectoryRestore2.root");
   gSystem->Unlink("gdirectoryRestore3.root");

   return 0;
}
