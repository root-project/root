{
// compile and run the test in smatrix directory
   gROOT->ProcessLine(".L testKalman.cxx+");
   int ret = testKalman(); 
   if (ret == 0)
      std::cout << "testKalman OK" << std::endl;
   else
      std::cerr << "testKalman  FAILED !" << std::endl;

   return 0;  // need to always return zero if checking log file 
}
