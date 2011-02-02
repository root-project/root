#include "TestClass.cc+"

void readBack() 
{
  cout << "******************************************" << std::endl;
  TFile* outputA = TFile::Open("a.root");
  TestClass* testA = (TestClass*) outputA->Get("testA");
  cout << "fBits settings A: 0x" << hex << testA->TestRefBits(0xFFFFFFFF) << endl;
  cout << "Readback for testA->GetRef()" << std::endl << testA->GetRef() << endl;
  cout << "******************************************" << std::endl;
}

void test1() 
{
 
   readBack(); 

}
