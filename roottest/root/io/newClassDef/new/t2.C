void t2() {
MyClass<int> a;
MyClass<const double*> b;
(MyClass<int               >    * ) 0;
}

void t3() {
space::Nested<const double*> c;
}

