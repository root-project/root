{
   // Compile and run the code of math/unuran/test directory 
   gROOT->ProcessLine(".L unuranMultiDim.cxx+");
   int ret = unuranMultiDim();
   if (ret == 0)
       std::cout << "unuranMultiDim: OK" << std::endl;
   else
       std::cerr << "unuranMultiDim:  FAILED !" << std::endl;
   return 0;  // need to always return zero if checking log file
}
