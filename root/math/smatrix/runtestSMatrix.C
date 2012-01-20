{
// compile and run the test in smatrix directory
   gROOT->ProcessLine(".L testSMatrix.cxx+");
   int ret = testSMatrix(); 
   if (ret == 0)
      std::cerr << "testSMatrix OK" << std::endl;
   else
      std::cerr << "testSMatrix  FAILED !" << std::endl;

   return 0;  // need to always return zero if checking log file 
}
