#include <ROOT/TTreeProcessorMT.hxx>
#include <TROOT.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>

void workload(TTreeReader &r)
{
   TTreeReaderArray<double> ra(r, "truthCaloPt");
   while (r.Next())
      ra.GetSize();
}

// This is a regression test for https://github.com/root-project/root/issues/9136 and https://github.com/root-project/root/issues/10357 .
// Both issues require that the input trees contain a branch of type vector<T, RAdoptAllocator<T>> (or anyway a vector
// with a custom allocator, so that TTreeReaderArray uses emulated collection proxies).
int main()
{
   ROOT::EnableImplicitMT(2);
   ROOT::TTreeProcessorMT mt({"treeprocmt_race_regression_input1.root", "treeprocmt_race_regression_input2.root",
                              "treeprocmt_race_regression_input3.root", "treeprocmt_race_regression_input4.root"},
                             "t");
   mt.Process(workload);
}
