#include "RQ_OBJECT.h"
#include "Riostream.h"

class MyBaseClass : public TQObject {
   RQ_OBJECT("MyBaseClass")
private:
   Int_t fValue;
public:
   MyBaseClass() { fValue=0; }
   virtual ~MyBaseClass() { }
   Int_t GetValue() const { return fValue; }
   void SetValue(Int_t); // *SIGNAL*
};

void MyBaseClass::SetValue(Int_t v)
{
   if (v != fValue) {
      fValue = v;
      Emit("SetValue(Int_t)", v);
      cout << "value set to " << v << endl;
   }
}

class MyChildClass : public MyBaseClass {
public:
   MyChildClass() { }
   virtual ~MyChildClass() { }
};

class MyClass : public MyChildClass {
public:
   MyClass() { }
   virtual ~MyClass() { }
};

void runSignalSlots()
{
   MyClass *objA = new MyClass();
   MyClass *objB = new MyClass();
   objA->Connect("SetValue(Int_t)", "MyClass", objB, "SetValue(Int_t)");
   objB->SetValue(11);
   cout << "A: " << objA->GetValue() << endl;
   cout << "B: " << objB->GetValue() << endl;
   objA->SetValue(79);
   cout << "A: " << objA->GetValue() << endl;
   cout << "B: " << objB->GetValue() << endl;
}

