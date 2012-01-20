{
// compile and run the testGenVector in genvector directory
   gROOT->ProcessLine(".L testGenVector.cxx+");
   int ret = testGenVector(); 
   if (ret == 0)
      std::cout << "testGenVector OK" << std::endl;
   else
      std::cerr << "testGenVector  FAILED !" << std::endl;

   return 0;  // need to always return zero if checking log file 
}
