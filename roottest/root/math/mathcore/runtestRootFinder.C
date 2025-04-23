{
   // Compile and run the code of math/mathcore/test directory
   gROOT->ProcessLine(".L testRootFinder.cxx+");
#if defined(ClingWorkAroundMissingDynamicScope)
   int ret = 0;
   ret = gROOT->ProcessLine("testRootFinder();");
#else
  int ret = testRootFinder();
#endif
   if (ret == 0)
      std::cout << "testRooFinder OK" << std::endl;
   else
      std::cerr << "testRootFinder  FAILED !" << std::endl;

#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 0;
#else
      return 0;  // need to always return zero if checking log file  
#endif
}
