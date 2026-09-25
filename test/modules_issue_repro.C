#include "ROOT/RDataFrame.hxx"

int modules_issue_repro()
{
   // Coes doesn't even need to be executed
   if (false) {
      ROOT::RDataFrame df("t", "in.root");
      df.Snapshot("t", "out.root", {"b"});
   }
   return 0;
}
