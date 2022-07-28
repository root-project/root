/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOT_ROOFIT_MultiProcess_Config
#define ROOT_ROOFIT_MultiProcess_Config

#include <cstddef>  // std::size_t

namespace RooFit {
namespace MultiProcess {

class Config {
public:
   static void setDefaultNWorkers(unsigned int N_workers);
   static unsigned int getDefaultNWorkers();

   struct LikelihoodJob {
      // magic values to indicate that the number of tasks will be set automatically
      constexpr static std::size_t automaticNEventTasks = 0;
      constexpr static std::size_t automaticNComponentTasks = 0;

      static std::size_t defaultNEventTasks;
      static std::size_t defaultNComponentTasks;
   };
private:
   static unsigned int defaultNWorkers_;
};

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_Config
