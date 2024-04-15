#ifdef __CINT__

#include "RConfigure.h"

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link off all namespaces;

#pragma link C++ nestedclass;

#ifdef R__HAS_DATAFRAME
// BDT inference
#pragma link C++ class TMVA::Experimental::RBDT<TMVA::Experimental::BranchlessForest<float>>+;
#pragma link C++ class TMVA::Experimental::RBDT<TMVA::Experimental::BranchlessJittedForest<float>>+;
#endif

// RTensor will have its own streamer function
#pragma link C++ class TMVA::Experimental::RTensor<float,std::vector<float>>-;

#endif