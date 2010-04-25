{
   // Compile and run the code of math/unuran/test directory 
   gROOT->ProcessLine(".L $ROOTSYS/math/unuran/test/unuranSimple.cxx+");
   int ret = unuranSimple();
   if (ret == 0)
       std::cout << "unuranSimple: OK" << std::endl;
   else
       std::cerr << "unuranSimple:  FAILED !" << std::endl;
   return 0;  // need to always return zero if checking log file
}
