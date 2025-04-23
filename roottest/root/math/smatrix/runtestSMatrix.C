{
// compile and run the test in smatrix directory
   gROOT->ProcessLine(".L testSMatrix.cxx+");
#if defined(ClingWorkAroundMissingDynamicScope)
   int ret = 0;
   ret = gROOT->ProcessLine("testSMatrix();");
#else
   int ret = testSMatrix(); 
#endif
   if (ret == 0)
      std::cerr << "testSMatrix OK" << std::endl;
   else
      std::cerr << "testSMatrix  FAILED !" << std::endl;

#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 0;
#else
      return 0;  // need to always return zero if checking log file  
#endif
}
