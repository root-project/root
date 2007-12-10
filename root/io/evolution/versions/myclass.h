class MyClass {
public:
#if VERSION==1
   int a;
#elif VERSION==2
   float a;
#elif VERSION==3
   double a;
#elif VERSION==4
   short a;
#elif VERSION==5
   long a;
#endif;
};

#include "TFile.h"

#ifndef __CINT__
#if VERSION==3
RootClassVersion(MyClass,3)
#endif

#if VERSION==4
// Intentional too low
RootClassVersion(MyClass,2)
#endif

#if VERSION==5
// Intentional too low
RootClassVersion(MyClass,1)
#endif

#endif


void write(const char *filename) 
{
   TFile f(filename,"RECREATE");
   MyClass *m = new MyClass;
   f.WriteObject(m,"obj");
   f.Write();
}

void write_what(const char*what) 
{
   write(Form("myclass%s.root",what));
}
