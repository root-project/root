#include <ROOT/TDataFrame.hxx>
#include <TFile.h>

using namespace ROOT::Experimental;

void writeHistNoInputFile()
{
   std::cout << "write hist no input file" << std::endl;
   TDataFrame d(1);
   auto counter = 0;
   auto dd = d.Define("x", [&counter]() { return counter++; });
   auto h = dd.Histo1D<int>("x");

   TFile f("gdirectoryRestore1.root", "RECREATE");
   h->SetDirectory(&f);
   h->Write();
}

void makeDataSet()
{
   std::cout << "make dataset" << std::endl;
   TDataFrame d(1);
   auto counter = 0;
   auto dd = d.Define("x", [&counter]() { return counter++; });
   dd.Snapshot<int>("t", "gdirectoryRestore2.root", {"x"});
}

int writeHistWithInputFile()
{
   std::cout << "write hist with input file" << std::endl;
   TDataFrame d("t", "gdirectoryRestore2.root");
   auto h = d.Histo1D<int>("x");
   TFile f("gdirectoryRestore3.root", "RECREATE");
   h->SetDirectory(&f);
   h->Write();
   return nullptr == gDirectory;
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
