#if defined(ClingWorkAroundMissingSmartInclude) || defined(ClingWorkAroundMissingDynamicScope)
#include "TestClass.cc"
#else
#include "TestClass.cc+"
#endif

void readBack() 
{
  cout << "******************************************" << std::endl;
  TFile* outputA = TFile::Open("a.root");
  TestClass* testA = (TestClass*) outputA->Get("testA");
#ifdef ClingWorkAroundJITandInline
  cout.setf(ios_base::hex, ios_base::basefield);
  cout << "fBits settings A: 0x" << testA->TestRefBits(0xFFFFFFFF) << endl;
#else
  cout << "fBits settings A: 0x" << hex << testA->TestRefBits(0xFFFFFFFF) << endl;
#endif
  cout << "Readback for testA->GetRef()" << std::endl << testA->GetRef() << endl;
  cout << "******************************************" << std::endl;
}

void test1() 
{
 
   readBack(); 

}
