#ifdef __CLING__
#ifndef R__USE_CXXMODULES
// The following is needed only when C++ modules are not used
// - namespace is needed to enable autoloading, based on the namespace name
#pragma link C++ namespace ROOT::Experimental::ML;
#pragma link C++ namespace ROOT::Experimental::Internal::ML;
#pragma link C++ class ROOT::Experimental::Internal::ML::RDataLoaderEngine;
#endif
#endif
