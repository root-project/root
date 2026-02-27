/// \file
/// \ingroup tutorial_dataframe
/// \notebook -nodraw
/// Usage of multithreading mode with random generators.
///
/// This example illustrates how to define functions that generate random numbers and use them in an RDataFrame
/// computation graph in a thread-safe way.
///
/// Using only one random number generator in an application running with ROOT::EnableImplicitMT() is a common pitfall.
/// This pitfall creates race conditions resulting in a distorted random distribution. In the example, this issue is
/// solved by creating one random number generator per RDataFrame processing slot, thus allowing for parallel and
/// thread-safe access. The example also illustrates the difference between non-deterministic and deterministic random
/// number generation.
///
/// \macro_code
/// \macro_image
/// \macro_output
///
/// \date February 2026
/// \author Bohdan Dudar (JGU Mainz), Fernando Hueso-González (IFIC, CSIC-UV), Vincenzo Eduardo Padulano (CERN)

#include <iostream>
#include <memory>
#include <TCanvas.h>
#include <ROOT/RDataFrame.hxx>

#include "df041_ThreadSafeRNG.hxx"

// Canvas that should survive the running of this macro
std::unique_ptr<TCanvas> myCanvas;

void df041_ThreadSafeRNG()
{
   myCanvas = std::make_unique<TCanvas>("myCanvas", "myCanvas", 1000, 500);
   myCanvas->Divide(3, 1);

   unsigned int nEntries{10000000};

   // 1. Single thread for reference
   auto df1 = ROOT::RDataFrame(nEntries).Define("x", GetNormallyDistributedNumberFromGlobalGenerator);
   auto h1 = df1.Histo1D({"h1", "Single thread (no MT)", 1000, -4, 4}, {"x"});
   myCanvas->cd(1);
   h1->DrawCopy();

   // 2. One generator per RDataFrame slot, with random_device seeding
   // Notes and Caveats:
   // - How many numbers are drawn from each generator is not deterministic
   //   and the result is not deterministic between runs.
   unsigned int nSlots{8};
   ROOT::EnableImplicitMT(nSlots);
   // Before running the RDataFrame computation graph, we reinitialize the generators (one per slot), so they can
   // be used accordingly during the execution.
   ReinitializeGenerators(nSlots);
   auto df2 = ROOT::RDataFrame(nEntries).DefineSlot("x", GetNormallyDistributedNumberPerSlotGenerator);
   auto h2 = df2.Histo1D({"h2", "Thread-safe (MT, non-deterministic)", 1000, -4, 4}, {"x"});
   myCanvas->cd(2);
   h2->DrawCopy();

   // 3. One generator per RDataFrame slot, with entry seeding
   // Notes and Caveats:
   // - With RDataFrame(INTEGER_NUMBER) constructor (as in the example),
   //   the result is deterministic and identical on every run
   // - With RDataFrame(TTree) constructor, the result is not guaranteed to be deterministic.
   //   To make it deterministic, use something from the dataset to act as the event identifier
   //   instead of rdfentry_, and use it as a seed.

   // Before running the RDataFrame computation graph, we reinitialize the generators (one per slot), so they can
   // be used accordingly during the execution.
   ReinitializeGenerators(nSlots);
   auto df3 =
      ROOT::RDataFrame(nEntries).DefineSlot("x", GetNormallyDistributedNumberPerSlotGeneratorForEntry, {"rdfentry_"});
   auto h3 = df3.Histo1D({"h3", "Thread-safe (MT, deterministic)", 1000, -4, 4}, {"x"});
   myCanvas->cd(3);
   h3->DrawCopy();

   std::cout << std::fixed << std::setprecision(3) << "Final distributions                : " << "Mean " << " +- "
             << "StdDev" << std::endl;
   std::cout << std::fixed << std::setprecision(3) << "Theoretical                        : " << "0.000" << " +- "
             << "1.000" << std::endl;
   std::cout << std::fixed << std::setprecision(3) << "Single thread (no MT)              : " << h1->GetMean() << " +- "
             << h1->GetStdDev() << std::endl;
   std::cout << std::fixed << std::setprecision(3) << "Thread-safe (MT, non-deterministic): " << h2->GetMean() << " +- "
             << h2->GetStdDev() << std::endl;
   std::cout << std::fixed << std::setprecision(3) << "Thread-safe (MT, deterministic)    : " << h3->GetMean() << " +- "
             << h3->GetStdDev() << std::endl;
}
