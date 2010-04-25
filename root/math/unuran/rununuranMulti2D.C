{
   // Compile and run the code of math/unuran/test directory 
   gROOT->ProcessLine(".L $ROOTSYS/math/unuran/test/unuranMulti2D.cxx+");
   int ret = unuranMulti2D();
   if (ret == 0)
       std::cout << "unuranMulti2D: OK" << std::endl;
   else
       std::cerr << "unuranMulti2D:  FAILED !" << std::endl;
   return 0;  // need to always return zero if checking log file
}
