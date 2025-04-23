{
// compile and run the  test in genvector directory
   gROOT->ProcessLine(".L rotationApplication.cxx+");
#ifdef ClingWorkAroundMissingDynamicScope
   int ret = gROOT->ProcessLine("rotationApplication();"); 
#else
   int ret = rotationApplication(); 
#endif
   if (ret == 0)
      std::cerr << "test rotationApplication: OK" << std::endl;
   else
      std::cerr << "test rotationApplication: FAILED !" << std::endl;
 
#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 0;
#else
      return 0;  // need to always return zero if checking log file  
#endif
}
