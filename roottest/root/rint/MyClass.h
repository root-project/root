class MyClass {
public:
   static int fgMember;
private:
   static int fgPrivateMember;
};
int MyClass::fgMember = 42;
int MyClass::fgPrivateMember = 17;
