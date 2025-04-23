{
// compile and run the coordinates3D test in genvector
   gROOT->ProcessLine(".L coordinates4D.cxx+");
#if defined(ClingWorkAroundMissingDynamicScope)
   int ret = 0;
   ret = gROOT->ProcessLine("coordinates4D();");
#else
   int ret = coordinates4D(); 
#endif
   if (ret == 0)
      std::cerr << "test coordinates4D:  OK" << std::endl;
   else
      std::cerr << "test coordinates4D:  FAILED !" << std::endl;
   
#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 0;
#else
      return 0;  // need to always return zero if checking log file  
#endif
}
