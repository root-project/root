{
// compile and run the coordinates3D test in genvector
   gROOT->ProcessLine(".L $ROOTSYS/math/genvector/test/coordinates4D.cxx+");
   int ret = coordinates4D(); 
   if (ret == 0)
      std::cout << "test coordinates4D OK" << std::endl;
   else
      std::cerr << "test coordinates4D  FAILED !" << std::endl;
   
   return ret; 
}
