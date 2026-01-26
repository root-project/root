/// \file
/// \ingroup tutorial_dataframe
/// \notebook -nodraw
/// Usage of multithreading mode with random generators.
///
/// This tutorial illustrates how to make a thread-safe program with thread-local random number engines.
/// Using only one random number generator in a ROOT::EnableImplicitMT() mode is a common pitfall.
/// This pitfall creates race conditions resulting in a distorted random distribution.
/// This example illustrates how to define thread-safe functions that generate random numbers and use them in an
/// RDataFrame computation graph.
///
/// \macro_code
/// \macro_image
/// \macro_output
///
/// \date 2025
/// \author Bohdan Dudar (JGU Mainz) and Fernando Hueso-González (IFIC, CSIC-UV)

#include <iostream>
#include <thread>
#include <random>
#include "TCanvas.h"
#include "ROOT/RDataFrame.hxx"

// Canvas that should survive the running of this macro:
TCanvas *c1;

std::random_device globalRandomDevice{};
std::mt19937 globalGenerator(globalRandomDevice());
std::normal_distribution<double> globalGaus(0., 1.);

double GetGlobalNormallyDistributedNumber()
{
   return globalGaus(globalGenerator);
}

double GetNormallyDistributedNumber()
{
   thread_local std::random_device rd{};
   thread_local std::mt19937 generator(rd());
   thread_local std::normal_distribution<double> gaus(0., 1.);
   return gaus(generator);
}

double GetNormallyDistributedNumberForEntry(unsigned long long entry)
{
   // We want to generate a random number distributed according to a normal distribution in a thread-safe way and such
   // that it is reproducible across different RDataFrame runs, i.e. given the same input to the generator it will
   // produce the same value. This is one way to do it. It assumes that the input argument represents a unique entry ID,
   // such that any thread processing an RDataFrame task will see this number once throughout the entire execution of
   // the computation graph
   thread_local std::mt19937 generator;
   thread_local std::normal_distribution<double> gaus(0., 1.);
   // Calling both `reset` and `seed` methods is fundamental here to ensure reproducibility: without them the same
   // generator could be seeded by a different entry (depending on which is the first entry ID seen by a thread) or
   // could be at a different step of the sequence (depending how many entries this particular thread is processing).
   // Alternatively, if both the generator and the distribution objects were recreated from scratch at every function
   // call (i.e. by removing the `thread_local` attribute), then the next two method calls would not be necessary, at
   // the cost of a possible performance degradation.
   gaus.reset();
   generator.seed(entry);
   return gaus(generator);
}

void df041_ThreadSafeRNG()
{
   c1 = new TCanvas("c1", "c1", 1000, 500);
   c1->Divide(3, 1);

   // 1. Single thread for reference
   auto df1 = ROOT::RDataFrame(10000000).Define("x", GetGlobalNormallyDistributedNumber);
   auto h1 = df1.Histo1D({"h1", "Single thread (no MT)", 1000, -4, 4}, {"x"});
   c1->cd(1);
   h1->DrawClone();

   // 2. thread_local generators with random_device seeding
   // Notes and Caveats:
   // - how many numbers are drawn from each generator is not deterministic
   //   and the result is not deterministic between runs
   //   even if one seeded each generator with its rdfslot index.
   ROOT::EnableImplicitMT(32);
   auto df2 = ROOT::RDataFrame(10000000).Define("x", GetNormallyDistributedNumber);
   auto h2 = df2.Histo1D({"h2", "Thread-safe (MT, non-deterministic)", 1000, -4, 4}, {"x"});
   c1->cd(2);
   h2->DrawClone();

   // 3. thread_local generators with entry seeding
   // Notes and Caveats:
   // - With RDataFrame(INTEGER_NUMBER) constructor (as in the example),
   //   the result is deterministic and identical on every run
   // - With RDataFrame(TTree) constructor, the result is not guaranteed to be deterministic.
   //   To make it deterministic, use something from the dataset to act as the event identifier
   //   instead of rdfentry_, and use it as a seed.

   auto df3 = ROOT::RDataFrame(10000000).Define("x", GetNormallyDistributedNumberForEntry, {"rdfentry_"});
   auto h3 = df3.Histo1D({"h3", "Thread-safe (MT, deterministic)", 1000, -4, 4}, {"x"});
   c1->cd(3);
   h3->DrawClone();

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
