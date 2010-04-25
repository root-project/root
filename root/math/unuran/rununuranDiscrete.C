{
   // Compile and run the code of math/unuran/test directory 
   gROOT->ProcessLine(".L $ROOTSYS/math/unuran/test/unuranDiscrete.cxx+");
   int ret = unuranDiscrete();
   if (ret == 0)
       std::cout << "unuranDiscrete: OK" << std::endl;
   else
       std::cerr << "unuranDiscrete:  FAILED !" << std::endl;
   return 0;  // need to always return zero if checking log file
}
