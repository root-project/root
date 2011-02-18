{
   // Compile and run the code of math/unuran/test directory 
   gROOT->ProcessLine(".L unuranDistr.cxx+");
   int ret = unuranDistr();
   if (ret == 0)
       std::cout << "unuranDistr: OK" << std::endl;
   else
       std::cerr << "unuranDistr:  FAILED !" << std::endl;
   return 0;  // need to always return zero if checking log file
}
