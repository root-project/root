#if defined(VERSION_ONE)
class Tdata {
   int myvar_one;
public:
   Tdata() : myvar_one(0) {}
   int get() { return myvar_one; }
   virtual ~Tdata() {}
   ClassDef(Tdata,1);   
};
class data {
   int myvar_one;
public:
   data() : myvar_one(0) {}
   int get() { return myvar_one; }
};
#elif defined(VERSION_TWO)
class Tdata {
   double mydouble_two;
public:
   Tdata() : mydouble_two(0) {}
   double get() { return mydouble_two; }
   virtual ~Tdata() {}
   ClassDef(Tdata,1);   
};
class data {
   double mydouble_two;
public:
   data() : mydouble_two(0) {}
   double get() { return mydouble_two; }
};
#elif defined(VERSION_THREE) || defined (VERSION_FIVE)
class Tdata {
   double mydouble;
   int more;
   int three;
public:
   Tdata() : mydouble(0),more(0),three(0) {}
   double get() { return more*three*mydouble; }
   virtual ~Tdata() {}
#if defined (VERSION_FIVE)
   ClassDef(Tdata,2);
#else
   ClassDef(Tdata,1);
#endif
};
class data {
   double mydouble;
   int more;
   int three;
public:
   data() : mydouble(0),more(0),three(0) {}
   double get() { return more*three*mydouble; }
};
#elif defined(VERSION_FOUR) || defined (VERSION_SIX)
class Tdata {
   double mydouble;
   double more;
   int four;
public:
   Tdata() : mydouble(0),more(0),four(0) {}
   double get() { return more*four*mydouble; }
   virtual ~Tdata() {}
#if defined (VERSION_SIX)
   ClassDef(Tdata,2);
#else
   ClassDef(Tdata,1);
#endif
};
class data {
   double mydouble;
   double more;
   int four;
public:
   data() : mydouble(0),more(0),four(0) {}
   double get() { return more*four*mydouble; }
};
#else
#error missing case
#endif

#ifdef __MAKECINT__
#pragma link C++ class data+;
#pragma link C++ class Tdata+;
#endif
