{
// compile and run the coordinates3D test in genvector directory
   gROOT->ProcessLine(".L coordinates3D.cxx+");
#if defined(ClingWorkAroundMissingDynamicScope)
   int ret = 0;
   ret = gROOT->ProcessLine("coordinates3D();");
#else
  int ret = coordinates3D();
#endif
   if (ret == 0)
      std::cerr << "test coordinates3D:  OK" << std::endl;
   else
      std::cerr << "test coordinates3D:  FAILED !" << std::endl;

#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 0;
#else
      return 0;  // need to always return zero if checking log file  
#endif
}
