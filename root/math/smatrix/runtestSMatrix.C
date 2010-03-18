{
// compile and run the test in smatrix directory
   gROOT->ProcessLine(".L $ROOTSYS/math/smatrix/test/testSMatrix.cxx+");
   int ret = testSMatrix(); 
   if (ret == 0)
      std::cout << "testSMatrix OK" << std::endl;
   else
      std::cerr << "testSMatrix  FAILED !" << std::endl;

   return 0;  // need to always return zero if checking log file 
}
