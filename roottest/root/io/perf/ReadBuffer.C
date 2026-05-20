#include "TBufferFile.h"
#include "TClass.h"
#include "TROOT.h"

class allint {
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
   allint() : a1(0),a2(0),a3(0),a4(0),a5(0),a6(0),a7(0),a8(0),a9(0) {}
};

class fltint {
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
   fltint() : a1(0),a2(0),a3(0),a4(0),a5(0),a6(0),a7(0),a8(0),a9(0) {}

};


void ReadBufferInt(int siz=10) {

   TBufferFile b(TBuffer::kWrite,32000);

   allint obj;
   TClass *cl = gROOT->GetClass(typeid(obj));
   cl->Streamer(&obj,b);

   b.SetReadMode();

   for(int i=0; i<siz; ++i) {
      b.Reset();
      cl->Streamer(&obj,b);
   }

}

void ReadBufferFloat(int siz=10) {

   TBufferFile b(TBuffer::kWrite,32000);

   fltint obj;
   TClass *cl = gROOT->GetClass(typeid(obj));
   cl->Streamer(&obj,b);

   b.SetReadMode();

   for(int i=0; i<siz; ++i) {
      b.Reset();
      cl->Streamer(&obj,b);
   }

}

#ifdef __ROOTCLING__
#pragma link C++ class allint+;
#pragma link C++ class fltint+;
#pragma link C++ function ReadBuffer;
#endif

int ReadBuffer(int kase = 1, int z = 1000000)
{
   switch (kase) {
      case 1: ReadBufferInt(z); break;
      case 2: ReadBufferFloat(z); break;
   }

   return 0;
}
