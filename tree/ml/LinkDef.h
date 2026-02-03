#ifdef __CLING__
#ifndef R__USE_CXXMODULES
// The following is needed only when C++ modules are not used
// - namespace is needed to enable autoloading, based on the namespace name
// - dictionary request for class template instantiation is needed to allow cppyy to request instantiation of any other
//   variation of the template. It leads to forward declaring the class template both in the generated rootmap and the
//   corresponding dictionary source file, and apparently only the second one is necessary for cppyy.
#pragma link C++ namespace ROOT::Experimental::ML;
#pragma link C++ namespace ROOT::Experimental::Internal::ML;
#pragma link C++ class ROOT::Experimental::Internal::ML::RBatchGenerator<int>;
#endif
#endif
