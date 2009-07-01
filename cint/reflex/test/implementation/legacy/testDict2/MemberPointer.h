#ifndef DICT2_MEMBERPOINTER_H
#define DICT2_MEMBERPOINTER_H


struct MyClass {
   void
   something(bool) {}

   int data;
};

struct MemberPointerTest {
   void
   setMemberPointer(int MyClass::* p) { fPtr = p; }

   int MyClass::*
   getMemberPointer() { return fPtr; }

   int MyClass::* fPtr;

   void
   setFunctionMemberPointer(void (MyClass::* p)(bool)) { fFptr = p; }

   void(MyClass::* getFunctionMemberPointer()) (bool) { return fFptr; }
   void (MyClass::* fFptr)(bool);
};

#endif // DICT2_FUNCTIONS_H
