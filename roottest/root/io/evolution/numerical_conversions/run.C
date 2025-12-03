void run(const char *what) {
   if (0 == compile(what))
#if defined(ClingWorkAroundMissingDynamicScope) || defined(__CLING__)
      gROOT->ProcessLine("write();");
#else
      write();
#endif
}
