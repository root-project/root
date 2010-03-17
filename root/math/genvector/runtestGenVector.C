{
// compile and run the testGenVector in genvector directory
   gErrorIgnoreLevel = kInfo+1;  // to suppress Info messages from AClic
   gROOT->ProcessLine(".L $ROOTSYS/math/genvector/test/testGenVector.cxx+");
   int ret = testGenVector(); 
   if (ret == 0)
      std::cout << "testGenVector OK" << std::endl;
   else
      std::cerr << "testGenVector  FAILED !" << std::endl;
   
}
