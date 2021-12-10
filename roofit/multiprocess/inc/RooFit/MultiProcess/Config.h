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

namespace RooFit {
namespace MultiProcess {

class Config {
public:
   static void setDefaultNWorkers(unsigned int N_workers);
   static unsigned int getDefaultNWorkers();
private:
   static unsigned int defaultNWorkers_;
};

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_Config
