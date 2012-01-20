{
// compile and run the test in smatrix directory
   gROOT->ProcessLine(".L testOperations.cxx+");
   int ret = testOperations(); 
   if (ret == 0)
      std::cout << "testOperations OK" << std::endl;
   else
      std::cerr << "testOperations  FAILED !" << std::endl;

   return 0;  // need to always return zero if checking log file  
}
