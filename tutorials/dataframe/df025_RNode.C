/// \file
/// \ingroup tutorial_dataframe
/// \notebook
/// RNode is a generic type which represents any transformation node in the computation graph.
/// This tutorial shows how to take advantage of the RNode class.
///
/// \macro_code
/// \macro_output
///
/// \date December 2018
/// \author Danilo Piparo

/// This function does not need to be a template: the RNode type accommodates all
/// possible nodes.
ROOT::RDF::RNode AddFilter(ROOT::RDF::RNode node, string_view filterStr)
{
   return node.Filter(filterStr);
}

/// Trivial helper function which returns the demangled typename from a typeid
template<typename T>
std::string GetName(T&)
{
   int dummy;
   return TClassEdit::DemangleName(typeid(T).name(), dummy);
}

void df025_RNode()
{
   ROOT::RDataFrame df(8);
   std::cout << "Type name of input node: " << GetName(df) << std::endl;
   auto f1 = AddFilter(df, "rdfentry_ > 0");
   auto f2 = f1.Filter([](ULong64_t e) { return e > 1; }, {"rdfentry_"});
   std::cout << "Type name of input node: " << GetName(f2) << std::endl;
   auto f3 = AddFilter(f2, "rdfentry_ > 2");

   std::cout << "Entries passing the selection: " << *f3.Count() << std::endl;
}
