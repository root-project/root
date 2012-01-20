{
// compile and run the test in smatrix directory
   gROOT->ProcessLine(".L testInversion.cxx+");
   int ret = testInversion(); 
   if (ret == 0)
      std::cout << "testInversion OK" << std::endl;
   else
      std::cerr << "testInversion  FAILED !" << std::endl;

   return 0;  // need to always return zero if checking log file  
}
