#ifndef refClasses_cxx
#define refClasses_cxx

#include <stdio.h>
#include "TString.h"

class yy {
public:
TString value;
   yy(const char *arg) { value = arg; fprintf(stdout,"creating a yy with %s\n",arg); };
  ~yy() { fprintf(stdout,"deleting a yy with %s\n",value.Data()); }
  const char *Data() const { return value.Data(); }
};

class zz {
public:
TString value;
   zz(const yy &arg) : value(arg.Data()) { fprintf(stdout,"Copying yy into zz with %s\n",arg.Data()); }
   zz(const char *arg) { value = arg; fprintf(stdout,"creating a zz with %s\n",arg); };
  ~zz() { fprintf(stdout,"deleting a zz with %s\n",value.Data()); }
  const char *Data() { return value.Data(); }
};

#endif

