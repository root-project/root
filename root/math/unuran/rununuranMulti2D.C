{
   // Compile and run the code of math/unuran/test directory 
   gROOT->ProcessLine(".L unuranMulti2D.cxx+");
#if defined(ClingWorkAroundUnnamedIncorrectInitOrder) || \
   defined(ClingWorkAroundMissingDynamicScope)
   int ret = 0;
   ret = gROOT->ProcessLine("unuranMulti2D();");
#else
   int ret = unuranMulti2D();
#endif
   if (ret == 0)
       std::cout << "unuranMulti2D: OK" << std::endl;
   else
       std::cerr << "unuranMulti2D:  FAILED !" << std::endl;

#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 0;
#else
      return 0;  // need to always return zero if checking log file  
#endif
}
