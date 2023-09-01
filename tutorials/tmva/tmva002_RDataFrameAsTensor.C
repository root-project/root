/// \file
/// \ingroup tutorial_tmva
/// \notebook -nodraw
/// This tutorial shows how the content of an RDataFrame can be converted to an
/// RTensor object.
///
/// \macro_code
/// \macro_output
///
/// \date December 2018
/// \author Stefan Wunsch

using namespace TMVA::Experimental;

void tmva002_RDataFrameAsTensor()
{
   // Creation of an RDataFrame with five entries filled with ascending numbers
   ROOT::RDataFrame df(5);
   auto df2 = df.Define("x", "1.f*rdfentry_").Define("y", "-1.f*rdfentry_");

   // Convert content of columns to an RTensor object
   auto x = AsTensor<float>(df2);

   std::cout << "RTensor from an RDataFrame:\n" << x << "\n\n";

   // The utility also supports reading only a part of the RDataFrame and different
   // memory layouts.
   auto x2 = AsTensor<float>(df2, {"x"}, MemoryLayout::ColumnMajor);

   std::cout << "RTensor from a single column of the RDataFrame:\n" << x2 << "\n\n";
}
