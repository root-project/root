#include "TObject.h"
#include "vector"

template <class T> class MyTemplate : public TObject {
 public:
  T variable; //!
  std::vector<int> vars;
  
  MyTemplate(T a) { variable = a; };
  MyTemplate() {};

  ClassDefT(MyTemplate,1)
};

ClassDefT2(MyTemplate,T)

     //MyTemplate<const int*> dummy;

template <>
class MyTemplate <const double*> : public TObject {
 public:
  const double* variable; //!
  std::vector<int> vars;
  
  MyTemplate<const double*>(const double* a) { variable = a; };
  MyTemplate<const double*>() {};
  
  ClassDef(MyTemplate<const double*>,2)
};

template <class T1, class T2> class MyPairTemplate : public TObject {
 public:
  T1 var1;
  T2 var2;
  
  MyPairTemplate(T1 a, T2 b) : var1(a), var2(b) {};
  ~MyPairTemplate() {};

  ClassDefT(MyPairTemplate,1)
};

ClassDef2T2(MyPairTemplate,T1,T2)


#if 0
// Can't do 2 arguments template specialization yet.....
template <> 
class MyPairTemplate<int, double> : public TObject {
 public:
  float var1;
  float var2;
  
  MyPairTemplate<int,double>(int a, double b) : var1(a), var2(b) {};
  ~MyPairTemplate<int,double>() {};

#ifdef __CINT__
  ClassDef(MyPairTemplate<int,double>,2)
#else
  ClassDef(MyPairTemplate,2)
#endif
};
#endif

template <class RootClass> class R__tInit {
public:
   R__tInit(Int_t pragmabits) {
      AddClass(RootClass::Class_Name(),
               RootClass::Class_Version(),
               &RootClass::Dictionary,pragmabits);
   }
   static void SetImplFile(char *file, int line) {
     fgImplFileName = file;
     fgImplFileLine = line;
   }
   static char* fgImplFileName; // Need to be instantiated
   static int   fgImplFileLine;
   static char* GetImplFileName() { return fgImplFileName; }
   static int   GetImplFileLine() { return fgImplFileLine; }
   ~R__tInit() { RemoveClass(RootClass::Class_Name()); }

};

#if 0
template <template <typename T> class RootClass > class R__tInit1 {
public:
   R__tInit1(Int_t pragmabits) {
      AddClass(RootClass::Class_Name(),
               RootClass::Class_Version(),
               &RootClass::Dictionary,pragmabits);
   }
   static void SetImplFile(char *file, int line) {
     fgImplFileName = file;
     fgImplFileLine = line;
   }
   static char* fgImplFileName; // Need to be instantiated
   static int   fgImplFileLine;
   static char* GetImplFileName() { return fgImplFileName; }
   static int   GetImplFileLine() { return fgImplFileLine; }
   ~R__tInit1() { RemoveClass(RootClass::Class_Name()); }

};
#endif 

template <class T1, class T2> class R__Setter1 { \
public: \
  R__Setter1() { \
  } \
}; 

#define ClassImpTGeneric(name) \
template <class T1> class R__Setter1<T1,name<T1> > { \
public: \
  R__Setter1() { \
    R__tInit< name<T1> >::SetImplFile(__FILE__,__LINE__); \
  } \
}; 

#if 0
template <class T1, class T2, class T3> class R__Setter2 {
public:
  R__Setter2(const char* file, int line) {
    R__tInit<RootClass>::SetImplFile((char*)file,line);
  }
};

class R__Setter3 {
public:
  template <class RootClass> R__Setter3(const char* file, int line, RootClass* ) {
    R__tInit<RootClass>::SetImplFile((char*)file,line);
  }
};
#endif

template <class T> char * R__tInit<T >::fgImplFileName = __FILE__; 
template <class T> int R__tInit<T >::fgImplFileLine = __LINE__;

#define newClassImpT(name) \


//template <> char * R__tInit1<name  >::fgImplFileName = __FILE__; \
//template <> int R__tInit1<name >::fgImplFileLine = __LINE__;
  

#define optClassImpOld(name) \
namespace { static R__Setter<name> setter(__FILE__,__LINE__); };

#define optClassImpT(name) \
namespace { static R__Setter<name> setter(__FILE__,__LINE__); };

#define optClassImp(name) \
namespace { static R__Setter2 setter(__FILE__,__LINE__, (name *) 0x0 ); }


//newClassImpT(MyTemplate)


