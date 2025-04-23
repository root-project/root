{
   // Compile and run the code of math/unuran/test directory 
   gROOT->ProcessLine(".L unuranDiscrete.cxx+");
#if defined(ClingWorkAroundMissingDynamicScope)
   int ret = 0;
   ret = gROOT->ProcessLine("unuranDiscrete();");
#else
   int ret = unuranDiscrete();
#endif
   if (ret == 0)
       std::cout << "unuranDiscrete: OK" << std::endl;
   else
       std::cerr << "unuranDiscrete:  FAILED !" << std::endl;
#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 0;
#else
      return 0;  // need to always return zero if checking log file  
#endif
}
