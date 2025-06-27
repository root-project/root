#ifndef abstract_h
#define abstract_h

#include <cstdio>

class Abstract {
public:
   Int_t fId;

   Abstract(Int_t id = -1) : fId(id) {}

   virtual ~Abstract() { fprintf(stdout,"Running Abstract's destructor\n"); }
   virtual void Action() = 0;
};

#ifdef __MAKECINT__
#pragma link C++ class Abstract+;
#endif

#endif