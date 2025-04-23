#if defined(CASE_longlong)
#undef MYTYPE
#define MYTYPE long long
#endif

#include "float16.h"

void write() {
   TString filename( Form("%s.root", WHAT) );
   printf("writing %s\n",filename.Data());
   write(filename);
}

