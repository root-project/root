#include "TBuffer.h"
#include "TClass.h"
#include "TStreamerInfo.h"

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


void ReadBuffer(int siz=10) {

   TBuffer b(TBuffer::kWrite,32000);

   allint obj;
   TClass *cl = gROOT->GetClass(typeid(obj));
   cl->Streamer(&obj,b);

   b.SetReadMode();
   
   for(int i=0; i<siz; ++i) {
      b.Reset();
      cl->Streamer(&obj,b);
   }

}

void ReadBufferMix(int siz=10) {

   TBuffer b(TBuffer::kWrite,32000);

   fltint obj;
   TClass *cl = gROOT->GetClass(typeid(obj));
   cl->Streamer(&obj,b);
   
   b.SetReadMode();
   
   for(int i=0; i<siz; ++i) {
      b.Reset();
      cl->Streamer(&obj,b);
   }

}

void InfoReadBuffer(int siz=10) {

   TBuffer b(TBuffer::kWrite,32000);

   allint obj;
#if ROOT_VERSION_CODE<= 199169

#else
   char *pointer = (char*)&obj;
#endif
   TClass *cl = gROOT->GetClass(typeid(obj));
   TStreamerInfo *info = cl->GetStreamerInfo();

#if ROOT_VERSION_CODE<= 199169
      info->WriteBuffer(b, (char*)(&obj),-1);
#else
      info->WriteBufferAux(b, &pointer,-1, 1, 0, 0);
#endif
   
   b.SetReadMode();
   
   for(int i=0; i<siz; ++i) {
      b.Reset();
      // cl->Streamer(&obj,b);
#if ROOT_VERSION_CODE<= 199169
      info->ReadBuffer(b, (char*)(&obj),-1);
#else
      info->ReadBuffer(b, (char*)(&obj),-1, 1, 0, 0);
#endif
   }

}

void InfoReadBufferMix(int siz=10) {

   TBuffer b(TBuffer::kWrite,32000);

   fltint obj;
#if ROOT_VERSION_CODE<= 199169

#else
   char *pointer = (char*)&obj;
#endif
   TClass *cl = gROOT->GetClass(typeid(obj));
   TStreamerInfo *info = cl->GetStreamerInfo();
   
#if ROOT_VERSION_CODE<= 199169
      info->WriteBuffer(b, (char*)(&obj),-1);
#else
      info->WriteBufferAux(b, &pointer,-1, 1, 0, 0);
#endif
   
   b.SetReadMode();
   
   for(int i=0; i<siz; ++i) {
      b.Reset();
      // cl->Streamer(&obj,b);
#if ROOT_VERSION_CODE<= 199169
      info->ReadBuffer(b, (char*)(&obj),-1);
#else
      info->ReadBuffer(b, (char*)(&obj),-1, 1, 0, 0);
#endif
   }

}



#ifdef __MAKECINT__
#pragma link C++ class allint+;
#pragma link C++ class fltint+;
#pragma link C++ function ReadBuffer;
#endif

#ifndef __CINT__
int main(int argc,char**argv) {

   if (argc!=3) {
      fprintf(stderr,"ReadBuffer requires 1 argument:\n");
      fprintf(stderr,"ReadBuffer <test type> <samplesize>\n");
      return 1;
   }
   
   int kase = atoi(argv[1]);
   int z = atoi(argv[2]);

   switch (kase) {
      case 1: ReadBuffer(z); break;
      case 2: InfoReadBuffer(z); break; 
         
      case 11: ReadBufferMix(z); break;
      case 12: InfoReadBufferMix(z); break; 
         
   }

   return 0;
}
#endif
