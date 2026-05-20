int execwriteClassWithComplex(){
   classWithComplex c(1,2,3,4);
   classWithComplex cc(11,22,33,44);
   TFile f("classWithComplex.root","RECREATE");
   f.WriteObject(&c,"classWithComplex1");
   f.WriteObject(&cc,"classWithComplex2");

   TXMLFile x("classWithComplex.xml","RECREATE");
   x.WriteObject(&c,"classWithComplex1");
   x.WriteObject(&cc,"classWithComplex2");

   return 0;
}
