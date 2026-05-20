{
   // Load the official TClass.
   auto c = TClass::GetClass("vector<double>");
   // then Load a library containing a duplicate
   gROOT->ProcessLine(".L libDuplicate.cxx+");
   // Test the the list of data members content
   return testDuplicate();
}
