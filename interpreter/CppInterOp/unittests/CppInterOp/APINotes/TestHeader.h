#ifndef UNITTESTS_CPPINTEROP_APINOTES_TESTHEADER_H
#define UNITTESTS_CPPINTEROP_APINOTES_TESTHEADER_H

void* testAlloc(int value);
void testNotAlloc(void* ptr);

void* testMalloc();
void* testNew();
void* testNewArr();
void* testOperatorNew();
void* testOperatorNewArr();
void* testNone();
void* testWeirdAttr();

void testDeallocFree(void* ptr);
void testDeallocDelete(void* ptr);
void testDeallocDeleteArr(void* ptr);
void testDeallocNone(void* ptr);
void testDeallocMulti(void* a, void* b);
void testDeallocWeirdAttr(void* ptr);

class KlassNotes {
public:
  void releaseArg(void* ptr);
  void releaseSelf();
  void releaseBoth(void* ptr);
  void releaseBare(void* ptr);
};
#endif
