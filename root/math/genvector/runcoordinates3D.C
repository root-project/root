{
// compile and run the coordinates3D test in genvector directory
   gROOT->ProcessLine(".L $ROOTSYS/math/genvector/test/coordinates3D.cxx+");
   int ret = coordinates3D(); 
   if (ret == 0)
      std::cout << "test coordinates3D OK" << std::endl;
   else
      std::cerr << "test coordinates3D  FAILED !" << std::endl;
   
   return ret; 
}
