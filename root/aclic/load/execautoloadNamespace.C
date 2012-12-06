{
#ifndef ClingWorkAroundMissingAutoLoading
MyNamespace::MyClass a;
#else
TClass::GetClass("MyNamespace::MyClass");
 int res = 0;
#endif

}
