/// \file
/// \ingroup tutorial_dataframe
/// \notebook -nodraw
/// Usage of RResultPtr.
///
/// This tutorial illustrates what is the difference between lazy and immediate results and how to use either of them in RDataFrame.
/// "Lazy" or deferred results are only produced once they are accessed. This allows for declaring multiple desired results, and producing
/// them in a single run of the event loop.
///
/// \macro_code
/// \macro_output
///
/// \date 2025
/// \author Stephan Hageboeck (CERN)

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>

#include <iostream>

// A function that adds a "lazy" histogram to a computation graph.
// The event loop will not run if only the RResultPtr is declared.
ROOT::RDF::RResultPtr<TH1D> histoLater(ROOT::RDF::RNode & rdf) {
   return rdf.Histo1D({"Histo2", "Histogram running later", 10, 0, 20}, {"x"});
}

// A function that immediately produces a result.
std::shared_ptr<TH1D> histoNow(ROOT::RDF::RNode & rdf) {
   auto histo = rdf.Histo1D({"Histo2", "Histogram running later", 10, 0, 20}, {"x"});
   return histo.GetSharedPtr();
}


void df039_RResultPtr_basics()
{
   // Create a simple dataframe that fills event numbers into a histogram.
   ROOT::RDataFrame bare_rdf(10);
   auto rdf = bare_rdf.Define("x", [&](unsigned long long entry) -> unsigned int {
         if (entry == 0) std::cout << "Event loop is running\n";
         return entry;
      }, {"rdfentry_"});

   // Book a histogram action. This will be stored as RResultPtr.
   // The action won't run yet.
   ROOT::RDF::RResultPtr<TH1D> histo1 = rdf.Histo1D({"Histo1", "Histogram", 10, 0, 10}, {"x"});
   std::cout << "Declared histo1\n";

   // When RDF results are declared in functions, one has to choose if one wants run it to immediately or lazily.
   // To run the event loop in a lazy fashion, return RResultPtr. This is equivalent to histo1 above, but happens
   // inside a function.
   auto rNode = ROOT::RDF::AsRNode(rdf);
   ROOT::RDF::RResultPtr<TH1D> histo2 = histoLater(rNode);
   std::cout << "Declared histo2\n";

   // If the function should produce the result immediately, a shared_ptr to the underlying result should be returned.
   std::shared_ptr<TH1D> histo3 = histoNow(rNode);
   std::cout << "Declared histo3\n";
}
