#ifndef ROOT_TUTORIALS_ANALYSIS_DATAFRAME_DF041
#define ROOT_TUTORIALS_ANALYSIS_DATAFRAME_DF041

#include <random>

// NOTE: these globals are intentionally NOT protected by a mutex.
// This function is only safe to call from a single thread (used as reference).
inline std::random_device globalRandomDevice{};
inline std::mt19937 globalGenerator(globalRandomDevice());
inline std::normal_distribution<double> globalGaus(0., 1.);

double GetNormallyDistributedNumberFromGlobalGenerator()
{
   return globalGaus(globalGenerator);
}

// One generator per slot — initialized once before the event loop
// An alternative to these global vectors could be to have thread_local
// variables within a function scope and then call that function from RDataFrame
inline std::vector<std::mt19937> generators;
inline std::vector<std::normal_distribution<double>> gaussians;

void ReinitializeGenerators(unsigned int nSlots)
{
   std::random_device rd;
   generators.resize(nSlots);
   for (auto &gen : generators)
      gen.seed(rd());
   gaussians.resize(nSlots, std::normal_distribution<double>(0., 1.));
}

double GetNormallyDistributedNumberPerSlotGenerator(unsigned int slot)
{
   return gaussians[slot](generators[slot]);
}

double GetNormallyDistributedNumberPerSlotGeneratorForEntry(unsigned int slot, unsigned long long entry)
{
   // We want to generate a random number distributed according to a normal distribution in a thread-safe way and such
   // that it is reproducible across different RDataFrame runs, i.e. given the same input to the generator it will
   // produce the same value. This is one way to do it. It assumes that the input argument represents a unique entry ID,
   // such that any thread processing an RDataFrame task will see this number once throughout the entire execution of
   // the computation graph
   // Calling both `reset` and `seed` methods is fundamental here to ensure reproducibility: without them the same
   // generator could be seeded by a different entry (depending on which is the first entry ID seen by a thread) or
   // could be at a different step of the sequence (depending how many entries this particular thread is processing).
   // Alternatively, if both the generator and the distribution objects were recreated from scratch at every function
   // call (i.e. by removing the global std::vector stores), then the next two method calls would not be necessary, at
   // the cost of a possible performance degradation.
   gaussians[slot].reset();
   generators[slot].seed(entry);
   return gaussians[slot](generators[slot]);
}

#endif // ROOT_TUTORIALS_ANALYSIS_DATAFRAME_DF041
