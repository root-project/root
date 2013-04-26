{
#ifndef ClingWorkAroundMissingAutoLoadingForNamespaces
MyNamespace::MyClass a;
#else
TClass::GetClass("MyNamespace::MyClass");
 int res = 0;
#endif

}
