{
// compile and run the test in smatrix directory
   gROOT->ProcessLine(".L $ROOTSYS/math/smatrix/test/testOperations.cxx+");
   int ret = testOperations(); 
   if (ret == 0)
      std::cout << "testOperations OK" << std::endl;
   else
      std::cerr << "testOperations  FAILED !" << std::endl;

   return ret; 
}
