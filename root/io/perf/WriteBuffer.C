#include "TBuffer.h"
#include "TClass.h"

class allints {
public:
   int a1;
   int a2;
   int a3;
   int a4;
   int a5;
   int a6;
   int a7;
   int a8;
   int a9;
   allints() : a1(0),a2(0),a3(0),a4(0),a5(0),a6(0),a7(0),a8(0),a9(0) {}
};

class floatint {
public:
   int a1;
   float a2;
   int a3;
   float a4;
   int a5;
   float a6;
   int a7;
   float a8;
   int a9;
   floatint() : a1(0),a2(0),a3(0),a4(0),a5(0),a6(0),a7(0),a8(0),a9(0) {}

};


void WriteBuffer(int siz=10) {

   TBuffer b(TBuffer::kWrite,32000);

   allints obj;
   TClass *cl = gROOT->GetClass(typeid(obj));
   
   for(int i=0; i<siz; ++i) {
      if (i % 1000000 == 0 ) b.Reset();
      cl->Streamer(&obj,b);
   }

}

#ifdef __MAKECINT__
#pragma link C++ class allints+;
#pragma link C++ class floatint+;
#pragma link C++ function WriteBuffer;
#endif

#ifndef __CINT__
int main(int argc,char**argv) {

   if (argc!=2) {
      fprintf(stderr,"WriteBuffer requires 1 argument:\n");
      fprintf(stderr,"WriteBuffer <samplesize>\n");
      return 1;
   }

   int z = atoi(argv[1]);

   WriteBuffer(z);

   return 0;
}
#endif
