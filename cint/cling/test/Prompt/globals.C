// RUN: cat %s | %cling | FileCheck %s
extern "C" int printf(const char*,...);
extern "C" void exit(int);

int i;
struct S{int i;} s;
i = 42;
extern "C" int printf(const char* fmt, ...);
printf("i=%d\n",i); // CHECK: i=42
if (i != 42) exit(1);
.q
