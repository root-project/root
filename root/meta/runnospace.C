{
// Fill out the code of the actual test
   gROOT->ProcessLine(".L WrapSimple.h+");
   if (gROOT->GetClass("AddSpace::Simple")==0) {
      cerr << "Could not retrive the class named AddSpace::Simple\n";
   }
   if (gROOT->GetClass("AddSpace")) {
      cerr << "We found the namespace AddSpace.  The test is not complete!\n";
   }
}
