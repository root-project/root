{
// compile and run the test in smatrix directory
   gROOT->ProcessLine(".L testInversion.cxx+");
#if defined(ClingWorkAroundMissingDynamicScope)
   int ret = 0;
   ret = gROOT->ProcessLine("testInversion();");
#else
   int ret = testInversion(); 
#endif
   if (ret == 0)
      std::cout << "testInversion OK" << std::endl;
   else
      std::cerr << "testInversion  FAILED !" << std::endl;


#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 0;
#else
      return 0;  // need to always return zero if checking log file  
#endif
}
