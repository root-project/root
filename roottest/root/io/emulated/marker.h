#ifndef marker_h
#define marker_h

#include <stdio.h>
#include "TString.h"

class Marker
{

   TString fClassName;

public:

   // Class to indicate whether a class's data member are
   // being properly destroyed.
   static Int_t fgDebug;

   Marker(TString name = "Marker") : fClassName(name) {}

   virtual ~Marker() {
      if (fgDebug>=2) fprintf(stdout, "Marker for %s 0x%lx\n", fClassName.Data(), (long)this);
      else if (fgDebug==1) fprintf(stdout, "Marker for %s\n", fClassName.Data());
   }

   ClassDef(Marker,2);
};

#ifdef __MAKECINT__
#pragma link C++ class Marker+;
#endif

Int_t Marker::fgDebug = 1;

#endif
