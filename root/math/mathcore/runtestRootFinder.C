{
   // Compile and run the code of math/mathcore/test directory
   gROOT->ProcessLine(".L testRootFinder.cxx+");
   int ret = testRootFinder(); 
   if (ret == 0)
      std::cout << "testRooFinder OK" << std::endl;
   else
      std::cerr << "testRootFinder  FAILED !" << std::endl;

   return 0;  // need to always return zero if checking log file 
}
