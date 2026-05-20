#ifdef __ROOTCLING__
#pragma link C++ class Inner<int>+;
#pragma link C++ class pair<string,Inner<int> >+;
#endif

