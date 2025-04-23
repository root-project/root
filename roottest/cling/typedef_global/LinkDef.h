#ifdef __CINT__
#pragma link off all    globals;
#pragma link off all    classes;
#pragma link off all    functions;


#pragma link C++ class ConstLink<Toy>;
#pragma link C++ class MyClass<Toy>-!;
#pragma link C++ nestedtypedef;
#pragma link C++ nestedclass;
#pragma link C++ typedef MyClass<Toy>::value_type;

#endif
