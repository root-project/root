#if defined(VERSION_ONE)
class Tdata {
   int myvar;
public:
   Tdata() : myvar(0) {}
   ClassDef(Tdata,1);   
};
class data {
   int myvar;
public:
   data() : myvar(0) {}
};
#elif defined(VERSION_TWO)
class Tdata {
   double mydouble;
public:
   Tdata() : mydouble(0) {}
   ClassDef(Tdata,1);   
};
class data {
   double mydouble;
public:
   data() : mydouble(0) {}
};
#elif defined(VERSION_THREE)
class Tdata {
   double mydouble;
   int more;
public:
   Tdata() : mydouble(0),more(0) {}
   ClassDef(Tdata,1);   
};
class data {
   double mydouble;
   int more;
public:
   data() : mydouble(0),more(0) {}
};
#elif defined(VERSION_FOUR)
class Tdata {
   double mydouble;
   double more;
public:
   Tdata() : mydouble(0),more(0) {}
   ClassDef(Tdata,1);   
};
class data {
   double mydouble;
   double more;
public:
   data() : mydouble(0),more(0) {}
};
#else
#error missing case
#endif

#ifdef __MAKECINT__
#pragma link C++ class data+;
#pragma link C++ class Tdata+;
#endif