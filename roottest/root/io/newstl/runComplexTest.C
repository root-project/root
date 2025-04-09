{
#include <complex>
   gROOT->ProcessLine(".L ComplexTest.h+");
   aTest = new Test();
   aTest->Set(3);
   if (!aTest->TestValue(3)) {
      cout << "The complex object was not set properly:\n";
      cout << aTest->fMyComplexVector.real() << ' ' << aTest->fMyComplexVector.imag() << '\n';
   } else {
      cout << "The complex object was set properly:\n";
      cout << aTest->fMyComplexVector.real() << ' ' << aTest->fMyComplexVector.imag() << '\n';
   }
   temp_file = new TFile("temp.root", "recreate");
   aTest->Write("test");
   temp_file->Close();
   
   Test *bTest = 0;
   temp_file = new TFile("temp.root", "read");
   temp_file->GetObject("test",bTest);
   if (!bTest->TestValue(3)) {
      cout << "The complex object was not read properly:\n";
      cout << bTest->fMyComplexVector.real() << ' ' << bTest->fMyComplexVector.imag() << '\n';
   } else {
      cout << "The complex object was read properly:\n";
      cout << bTest->fMyComplexVector.real() << ' ' << bTest->fMyComplexVector.imag() << '\n';
   }
      
   temp_file->Close();
}
