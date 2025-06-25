#if defined(ClingWorkAroundMissingSmartInclude) || defined(ClingWorkAroundMissingDynamicScope)
#include "TestClass.cc"
#else
#include "TestClass.cc+"
#endif
void readBack()
{
  TFile* outputB = TFile::Open("b.root");
  TestClass* testB = (TestClass*) outputB->Get("testB");
  if ( !testB ) { cout << "Couldn't find B?" << endl; return; }

#ifdef ClingWorkAroundJITandInline
   std::cout.setf(ios_base::hex, ios_base::basefield);
   std::cout << "fBits settings B: 0x" << testB->TestRefBits(0xFFFFFFFF) << std::endl;
#else
   std::cout << "fBits settings B: 0x" << hex << testB->TestRefBits(0xFFFFFFFF) << std::endl;
#endif

  std::cout << "Readback for testB->GetRef()" << std::endl << (ULongptr_t)testB->GetRef() << std::endl;
  std::cout << "******************************************" << std::endl;
}

void test2()
{
   readBack();
}
