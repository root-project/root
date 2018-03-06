#include <ROOT/TDataFrame.hxx>
#include <ROOT/TVec.hxx>
#include <TTree.h>
#include <TFile.h>

#include "test_read_leaves.h"

using namespace ROOT::Experimental;

int main()
{
   TDataFrame d("t", "test_read_leaves.root");
   d.Filter(check_a_b, {"v.a", "v.b"}).Report();
   return 0;
}

