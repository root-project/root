{
   // Compile and run the code of math/mathcore/test directory 
   gROOT->ProcessLine(".L testSpecFuncSiCi.cxx+");
#if defined(ClingWorkAroundMissingDynamicScope)
   int ret = 0;
   ret = gROOT->ProcessLine("testSpecFuncSiCi();");
#else
   int ret = testSpecFuncSiCi();
#endif
   if (ret == 0)
       std::cout << "testSpecFuncSiCi: OK" << std::endl;
   else
       std::cerr << "testSpecFuncSiCi:  FAILED !" << std::endl;

#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 0;
#else
      return 0;  // need to always return zero if checking log file  
#endif
}
