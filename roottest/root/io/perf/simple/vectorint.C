#define var(x) int i##x; float f##x
#define udef(x) i##x(0),f##x(0.0)
#define def(x) i##x(x),f##x(x/3.0)

#include "TNamed.h"
#include "TVirtualStreamerInfo.h"
 
#include "TBufferFile.h"
#include "TClass.h"
#include <vector> 
#ifdef __MAKECINT__
// #pragma link C++ class vector<int>+;
#endif

void write(TBuffer &buf,int ntimes, int nelems) {
   vector<int> clones; // "simple");
   for(int e=0; e<nelems; ++e) {
     clones.push_back(e);
   }
   TClass *cl = TClass::GetClass(typeid(clones));
   buf.SetWriteMode();
   for(int i=0; i<ntimes; ++i) {
      buf.Reset();
      cl->Streamer(&clones,buf);
   }
}

void read(TBuffer &buf,int ntimes) {
   vector<int> clones;
   TClass *cl = TClass::GetClass(typeid(clones));
   buf.SetReadMode();
   for(int i=0; i<ntimes; ++i) {
      buf.Reset();
      cl->Streamer(&clones,buf);
      // obj->IsA()->Dump(obj);
      // delete obj;
   }
}


void vectorint(int nread = 2, int nelems = 600) {
   TVirtualStreamerInfo::SetStreamMemberWise(kFALSE);
   TBufferFile buf(TBuffer::kWrite);
   write(buf,1, nelems);
   read(buf,nread);
}
