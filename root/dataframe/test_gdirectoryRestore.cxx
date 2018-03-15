#include <ROOT/TDataFrame.hxx>
#include <TFile.h>

using namespace ROOT::Experimental;

void writeHistNoInputFile()
{
   std::cout << "write hist no input file" << std::endl;
   TDataFrame d(1);
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
   TDataFrame d(1);
   auto counter = 0;
   auto dd = d.Define("x", [&counter]() { return counter++; });
   dd.Snapshot<int>("t", "gdirectoryRestore2.root", {"x"});
}

void writeHistWithInputFile()
{
   std::cout << "write hist with input file" << std::endl;
   TDataFrame d("t", "gdirectoryRestore2.root");
   auto m = d.Histo1D<int>("x");
   TFile f("gdirectoryRestore3.root", "RECREATE");
   *m;
   assert(gDirectory == &f);
}

int test_gdirectoryRestore()
{
   ROOT::EnableImplicitMT();
   writeHistNoInputFile();
   makeDataSet();
   return writeHistWithInputFile();
}

int main()
{
   return test_gdirectoryRestore();
}
