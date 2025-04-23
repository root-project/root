{
// compile and run the test in smatrix directory
   gROOT->ProcessLine(".L testOperations.cxx+");
#if defined(ClingWorkAroundMissingDynamicScope)
   int ret = 0;
   ret = gROOT->ProcessLine("testOperations();");
#else
   int ret = testOperations();
#endif
   if (ret == 0)
      std::cout << "testOperations OK" << std::endl;
   else
      std::cerr << "testOperations  FAILED !" << std::endl;


#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 0;
#else
      return 0;  // need to always return zero if checking log file  
#endif
}
