{
// compile and run the test in smatrix directory
   gROOT->ProcessLine(".L $ROOTSYS/math/smatrix/test/testKalman.cxx+");
   int ret = testKalman(); 
   if (ret == 0)
      std::cout << "testKalman OK" << std::endl;
   else
      std::cerr << "testKalman  FAILED !" << std::endl;

   return 0;  // need to always return zero if checking log file 
}
