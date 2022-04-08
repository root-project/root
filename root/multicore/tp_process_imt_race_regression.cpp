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

int main()
{
   ROOT::EnableImplicitMT(2);
   ROOT::TTreeProcessorMT mt({"treeprocmt_race_regression_input1.root", "treeprocmt_race_regression_input2.root",
                              "treeprocmt_race_regression_input3.root", "treeprocmt_race_regression_input4.root"},
                             "t");
   mt.Process(workload);
}
