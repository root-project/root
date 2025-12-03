{
#ifndef ClingWorkAroundMissingAutoLoadingForNamespaces
MyNamespace::MyClass a;
#else
TClass::GetClass("MyNamespace::MyClass");
#endif
int res = 0;
}
