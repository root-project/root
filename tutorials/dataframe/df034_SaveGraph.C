/// \file
/// \ingroup tutorial_dataframe
/// \notebook -nodraw
/// Basic SaveGraph usage.
///
/// This tutorial shows how to use the SaveGraph action.
/// SaveGraph inspects the sequence of RDataFrame actions.
///
/// \macro_code
/// \macro_output
///
/// \date January 2022
/// \author Ivan Kabadzhov (CERN)

// First, an RDataFrame computation graph is created with Defines, Filters and methods such as Mean, Count, etc.
// After that, SaveGraph can be called either on the root RDataFrame object or on a specific node of the computation
// graph: in the first case, the graph returned will span the full computation graph, in the second case it will show
// only the branch of the computation graph that the node belongs to.
// If a filename is passed as second argument, the graph is saved to that file, otherwise it is returned as a string.

R__LOAD_LIBRARY(ROOTDataFrame) // only required for ROOT installations without runtime_cxxmodules

void df034_SaveGraph()
{
   ROOT::RDataFrame rd1(1);
   auto rd2 = rd1.Define("Root_def1", "1").Filter("Root_def1 < 2", "Main_Filter").Define("Root_def2", "1");

   auto branch1 = rd2.Define("Branch_1_def", "1");
   auto branch2 = rd2.Define("Branch_2_def", "1");

   ROOT::RDF::RResultPtr<double> branch1_1 = branch1.Filter("Branch_1_def < 2", "Filter_1")
                                                .Define("Branch_1_1_def", "1")
                                                .Filter("1 == Branch_1_1_def % 2", "Filter_1_1")
                                                .Mean("Branch_1_1_def");

   ROOT::RDF::RResultPtr<unsigned long long> branch1_2 =
      branch1.Define("Branch_1_2_def", "1").Filter("Branch_1_2_def < 2", "Filter_1_2").Count();

   ROOT::RDF::RResultPtr<double> branch2_1 = branch2.Filter("Branch_2_def < 2", "Filter_2")
                                                .Define("Branch_2_1_def", "1")
                                                .Define("Branch_2_2_def", "1")
                                                .Filter("1 == Branch_2_1_def % 2", "Filter_2_1")
                                                .Max("Branch_2_1_def");

   ROOT::RDF::RResultPtr<unsigned long long> branch2_2 = branch2.Count();

   std::cout << ROOT::RDF::SaveGraph(branch1_1);
   ROOT::RDF::SaveGraph(rd1, /*output_file=*/"rdf_savegraph_tutorial.dot");
   // SaveGraph produces content in the standard DOT file format
   // (https://en.wikipedia.org/wiki/DOT_%28graph_description_language%29): it can be converted to e.g. an image file
   // using standard tools such as the `dot` CLI program.
   gSystem->Exec("dot -Tpng rdf_savegraph_tutorial.dot -o rdf_savegraph_tutorial.png");
}
