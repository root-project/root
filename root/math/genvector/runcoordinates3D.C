{
// compile and run the coordinates3D test in genvector directory
   gROOT->ProcessLine(".L coordinates3D.cxx+");
   int ret = coordinates3D(); 
   if (ret == 0)
      std::cerr << "test coordinates3D:  OK" << std::endl;
   else
      std::cerr << "test coordinates3D:  FAILED !" << std::endl;
   
   return 0;  // need to always return zero if checking log file 
}
