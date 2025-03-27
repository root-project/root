/// \file
/// \ingroup tutorial_dataframe
/// \notebook -nodraw
/// Usage of RResultPtr: Lifetime management.
///
/// This tutorial illustrates how to manage the lifetime of RDataFrame results.
/// When RDataFrame results are declared in functions (or scopes in general), 
/// they are destroyed at the end of the scope.
/// To prevent this, one needs to copy the RResultPtr or obtain a copy of its
/// underlying shared_ptr.
///
/// \macro_code
/// \macro_output
///
/// \date 2025
/// \author Stephan Hageboeck (CERN)

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <TCanvas.h>
#include <THStack.h>
#include <random>
#include <vector>

// Canvas that should survive the running of this macro:
TCanvas *c1, *c2;

void df040_RResultPtr_lifetimeManagement()
{
   // Create a simple dataframe that fills random numbers into histograms.
   ROOT::RDataFrame bare_rdf(10);
   std::mt19937 generator{1};
   std::normal_distribution gaus{5., 1.};
   auto rdf = bare_rdf.Define("x", [&]() -> double {
         return gaus(generator);
      }, {});

   ROOT::RDF::TH1DModel histoModel{"Histo", "Histo;x", 10, 0, 10};

   // Keeping the results alive is vital when they are passed to other entities or when they are drawn.
   // Compare the following situations:

   // 1. The wrong way (the result ht is destroyed at the end of the loop body):
   THStack histStack1("histStack1", "Stacking result histograms (wrong way)");
   for(int i=0; i<2; i++) {
        auto ht = rdf.Histo1D(histoModel, {"x"});
        ht->SetFillColor(kBlue+i);
        histStack1.Add(ht.GetPtr()); // Wrong, this histogram will not survive
   }
   c1 = new TCanvas("c1", "THStack without obtaining a shared_ptr (wrong)");
   histStack1.DrawClone();
   c1->Draw();

   // 2. The right way: Results survive because we copy the shared_ptr:
   THStack histStack2("histStack2", "THStack with shared_ptr (correct way)");
   std::vector<std::shared_ptr<TH1D>> results;
   for(int i=0; i<2; i++) {
        auto ht = rdf.Histo1D(histoModel, {"x"});
        ht->SetFillColor(kBlue+2*i);
        histStack2.Add(ht.GetPtr());
        results.push_back(ht.GetSharedPtr()); // Makes the histogram survive
   }
   c2 = new TCanvas("c2", "Drawing with obtaining a shared_ptr (right)");
   histStack2.DrawClone();
   c2->Draw();
}