#include "TH2.h"

void func(TH2*,const char* , const char* ,
          int , Axis_t , Axis_t ,
          int , Axis_t , Axis_t ,
          const char*) {};

void func(TH1*,const char* , const char* ,
          int , Axis_t , Axis_t ,
          const char*) {};

class HistMan {
public:
   template<class THType>
    THType* Book(const char*name, const char* title, 
                 int nbinsx, Axis_t xmin, Axis_t xmax,
                 const char* =".") {
      func((THType*)0,name,title,nbinsx,xmin,xmax,"");
        //TH1* h = Adopt(path, new THType(name,title,nbinsx,xmin,xmax));
        //return dynamic_cast<THType*>(h);
      return 0;
    }
           
    template<class THType>
    THType* Book(const char*name, const char* title, 
                 int nbinsx, Axis_t xmin, Axis_t xmax,
                 int nbinsy, Axis_t ymin, Axis_t ymax,
                 const char* = ".") {
       func((THType*)0,name,title,nbinsx,xmin,xmax,nbinsy,ymin,ymax,"");
        //TH1* h= Adopt(path, new THType(name,title,nbinsx,xmin,xmax,
        //                               nbinsy,ymin,ymax));
        //return dynamic_cast<THType*>(h);
       return 0;
    }
   ~HistMan() {}

   template <class THType> THType *grab1(const char*) { return 0; };
   template <class THType> THType *grab2(const char*) { return 0; };
};

#ifdef __MAKECINT__

#pragma link C++ class HistMan;
#pragma link C++ function HistMan::Book<TH1D>(const char*, const char*, int, Axis_t, Axis_t, const char*);
#pragma link C++ function HistMan::Book<TH2D>(const char*, const char*, int, Axis_t, Axis_t, int, Axis_t, Axis_t,const char*);
#pragma link C++ function HistMan::grab1<TH1F>(const char*);
#pragma link C++ function HistMan::grab2<TH1F>;

#endif

void duplicate() {
   HistMan m;
   m.Book<TH1D>("","",3,2,1,"");
   m.Book<TH2D>("","",3,2,1,4,3,1,"");


}
