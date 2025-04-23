{
   // Compile and run the code of math/unuran/test directory 
   gROOT->ProcessLine(".L unuranSimple.cxx+");
#if defined(ClingWorkAroundMissingDynamicScope)
   int ret = 0;
   ret = gROOT->ProcessLine("unuranSimple();");
#else
   int ret = unuranSimple();
#endif
   if (ret == 0)
       std::cout << "unuranSimple: OK" << std::endl;
   else
       std::cerr << "unuranSimple:  FAILED !" << std::endl;

#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 0;
#else
      return 0;  // need to always return zero if checking log file  
#endif
}
