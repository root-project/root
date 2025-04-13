{
// compile and run the testGenVector in genvector directory
    gROOT->ProcessLine(".L testGenVector.cxx+");
#if defined(ClingWorkAroundMissingDynamicScope)
   int ret = 0;
   ret = gROOT->ProcessLine("testGenVector();");
#else
   int ret = testGenVector(); 
#endif
   if (ret == 0)
      std::cout << "testGenVector OK" << std::endl;
   else
      std::cerr << "testGenVector  FAILED !" << std::endl;


#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 0;
#else
      return 0;  // need to always return zero if checking log file  
#endif
}
