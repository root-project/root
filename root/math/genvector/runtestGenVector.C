{
// compile and run the testGenVector in genvector directory
   gROOT->ProcessLine(".L $ROOTSYS/math/genvector/test/testGenVector.cxx+");
   int ret = main(); 
   if (ret == 0)
      std::cout << "testGenVector OK" << std::endl;
   else
      std::cerr << "testGenVector  FAILED !" << std::endl;
   
}
