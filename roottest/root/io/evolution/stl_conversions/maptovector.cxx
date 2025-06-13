#include "maptovector.h"

void write() {
   TString filename( Form("%s.root", WHAT) );
   printf("writing %s\n",filename.Data());
   write(filename);
}

