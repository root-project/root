{
// compile and run the test in smatrix directory
   gROOT->ProcessLine(".L testKalman.cxx+");
#if defined(ClingWorkAroundMissingDynamicScope)
   int ret = 0;
   ret = gROOT->ProcessLine("testKalman();");
#else
   int ret = testKalman(); 
#endif
   if (ret == 0)
      std::cout << "testKalman OK" << std::endl;
   else
      std::cerr << "testKalman  FAILED !" << std::endl;


#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 0;
#else
      return 0;  // need to always return zero if checking log file  
#endif
}
