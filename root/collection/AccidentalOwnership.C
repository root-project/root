
class MyClass : public TObject {
   int &fValRef;
   void TearDown() { ++fValRef; }

public:
   static int fgValue;
   MyClass() : fValRef(fgValue) {}
   ~MyClass() { TearDown(); }
   void Clear(Option_t *) override{};
   ClassDefOverride(MyClass, 1)
};
int MyClass::fgValue = 0;

// A test for ROOT-7249
int AccidentalOwnership()
{
   TClonesArray arr1("MyClass");
   ((TCollection *)&arr1)->TCollection::SetOwner(kTRUE);
   arr1.ConstructedAt(0);
   arr1.Delete();
   if (1 == MyClass::fgValue && ((TCollection *)&arr1)->TCollection::IsOwner()) {
      return 0;
   }
   std::cerr << "Error: Either the destructor of MyClass was not invoked or the TObjArray lost memory of its ownership."
             << std::endl;
   return 1;
}