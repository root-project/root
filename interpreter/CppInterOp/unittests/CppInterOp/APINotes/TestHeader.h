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
#endif
