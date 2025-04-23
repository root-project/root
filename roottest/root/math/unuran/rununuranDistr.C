{
   // Compile and run the code of math/unuran/test directory 
   gROOT->ProcessLine(".L unuranDistr.cxx+");
#if defined(ClingWorkAroundMissingDynamicScope)
   int ret = 0;
   ret = gROOT->ProcessLine("unuranDistr();");
#else
   int ret = unuranDistr();
#endif
   if (ret == 0)
       std::cout << "unuranDistr: OK" << std::endl;
   else
       std::cerr << "unuranDistr:  FAILED !" << std::endl;

#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 0;
#else
      return 0;  // need to always return zero if checking log file  
#endif
}
