#include "TH2.h"

class HistMan {
public:
   template<class THType>
    THType* Book(const char* name, const char* title, 
                 int nbinsx, Axis_t xmin, Axis_t xmax,
                 const char* path=".") {
        //TH1* h = Adopt(path, new THType(name,title,nbinsx,xmin,xmax));
        //return dynamic_cast<THType*>(h);
        return 0;
    }
           
    template<class THType>
    THType* Book(const char* name, const char* title,
                 int nbinsx, Axis_t xmin, Axis_t xmax,
                 int nbinsy, Axis_t ymin, Axis_t ymax,
                 const char* path=".") {
        //TH1* h= Adopt(path, new THType(name,title,nbinsx,xmin,xmax,
        //                               nbinsy,ymin,ymax));
        //return dynamic_cast<THType*>(h);
       return 0;
    }
};

#ifdef __MAKECINT__

#pragma link C++ function HistMan::Book<TH1D>(const char*, const char*, int, Axis_t, Axis_t, const char*);
#pragma link C++ function HistMan::Book<TH2D>(const char*, const char*, int, Axis_t, Axis_t, int, Axis_t, Axis_t,const char*);

#endif

void tt() {
   HistMan m;
   m.Book<TH1D>("","",3,2,1,"");
   m.Book<TH2D>("","",3,2,1,4,3,1,"");


}
