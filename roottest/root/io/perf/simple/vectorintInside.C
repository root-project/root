#define var(x) int i##x; float f##x
#define udef(x) i##x(0),f##x(0.0)
#define def(x) i##x(x),f##x(x/3.0)

#include "TNamed.h"
#include "TVirtualStreamerInfo.h"
#include <vector>

class Holder {
public:
   vector<int> fVector;
   ClassDef(Holder,2);
};

#include "TBufferFile.h"
#include "TClass.h"
#include <vector> 
#ifdef __MAKECINT__
// #pragma link C++ class vector<int>+;
#endif

void write(TBuffer &buf,int ntimes, int nelems) {
   Holder holder;
   for(int e=0; e<nelems; ++e) {
     holder.fVector.push_back(e);
   }
   buf.SetWriteMode();
   for(int i=0; i<ntimes; ++i) {
      buf.Reset();
      holder.Streamer(buf);
   }
}

void read(TBuffer &buf,int ntimes) {
   Holder holder;
   buf.SetReadMode();
   for(int i=0; i<ntimes; ++i) {
      buf.Reset();
      holder.Streamer(buf);
      // obj->IsA()->Dump(obj);
      // delete obj;
   }
}


void vectorintInside(int nread = 2, int nelems = 600) {
   TVirtualStreamerInfo::SetStreamMemberWise(kTRUE);
   TBufferFile buf(TBuffer::kWrite);
   write(buf,1, nelems);
   read(buf,nread);
}
