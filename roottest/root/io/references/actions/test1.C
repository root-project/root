#if defined(ClingWorkAroundMissingSmartInclude) || defined(ClingWorkAroundMissingDynamicScope)
#include "TestClass.cc"
#else
#include "TestClass.cc+"
#endif

void readBack()
{
  std::cout << "******************************************" << std::endl;
  TFile* outputA = TFile::Open("a.root");
  TestClass* testA = (TestClass*) outputA->Get("testA");
#ifdef ClingWorkAroundJITandInline
  std::cout.setf(ios_base::hex, ios_base::basefield);
  std::cout << "fBits settings A: 0x" << testA->TestRefBits(0xFFFFFFFF) << std::endl;
#else
  std::cout << "fBits settings A: 0x" << hex << testA->TestRefBits(0xFFFFFFFF) << std::endl;
#endif
  std::cout << "Readback for testA->GetRef()" << std::endl << (ULongptr_t)testA->GetRef() << std::endl;
  std::cout << "******************************************" << std::endl;
}

void test1()
{
  readBack();
}
