{
   // Compile and run the code of math/unuran/test directory 
   gROOT->ProcessLine(".L unuranMultiDim.cxx+");
#if defined(ClingWorkAroundMissingDynamicScope)
   int ret = 0;
   ret = gROOT->ProcessLine("unuranMultiDim();");
#else
  int ret = unuranMultiDim();
#endif
   if (ret == 0)
       std::cout << "unuranMultiDim: OK" << std::endl;
   else
       std::cerr << "unuranMultiDim:  FAILED !" << std::endl;

#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 0;
#else
      return 0;  // need to always return zero if checking log file  
#endif
}
