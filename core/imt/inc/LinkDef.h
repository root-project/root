#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// Only for the autoload, autoparse. No IO of these classes is foreseen!
#pragma link C++ class ROOT::Internal::TPoolManager-;
#pragma link C++ class ROOT::TThreadExecutor-;

#endif
