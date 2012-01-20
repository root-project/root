{
// compile and run the  test in genvector directory
   gROOT->ProcessLine(".L rotationApplication.cxx+");
   int ret = rotationApplication(); 
   if (ret == 0)
      std::cerr << "test rotationApplication: OK" << std::endl;
   else
      std::cerr << "test rotationApplication: FAILED !" << std::endl;
 
   return 0;  // need to always return zero if checking log file  
}
