#define var(x) int i##x; float f##x
#define udef(x) i##x(0),f##x(0.0)
#define def(x) i##x(x),f##x(x/3.0)

#include "TNamed.h"
#include "TVirtualStreamerInfo.h"
 
class simple : public TNamed {
private:
   var(0);
   var(1);
   var(2);
   var(3);
   var(4);
   var(5);
   var(6);
   var(7);
   var(8);
   var(9);

public:
   simple() :
      udef(0),udef(1),udef(2),udef(3),udef(4),udef(5),udef(6),
      udef(7),udef(8),udef(9)
   {}
   simple(int) : 
      def(0),def(1),def(2),def(3),def(4),def(5),def(6),
      def(7),def(8),def(9)
   {}

   ClassDef(simple,2);
};

#include "TBufferFile.h"
#include "TClass.h"
#include <vector> 
#ifdef __MAKECINT__
#pragma link C++ class vector<simple>+;
#endif

void write(TBuffer &buf,int ntimes, int nelems) {
   vector<simple> clones; // "simple");
   for(int e=0; e<nelems; ++e) {
     clones.push_back(simple(e));
   }
   TClass *cl = TClass::GetClass(typeid(clones));
   buf.SetWriteMode();
   for(int i=0; i<ntimes; ++i) {
      buf.Reset();
      cl->Streamer(&clones,buf);
   }
}

void read(TBuffer &buf,int ntimes) {
   vector<simple> clones;
   TClass *cl = TClass::GetClass(typeid(clones));
   buf.SetReadMode();
   for(int i=0; i<ntimes; ++i) {
      buf.Reset();
      cl->Streamer(&clones,buf);
      // obj->IsA()->Dump(obj);
      // delete obj;
   }
}


void vectortclass(int nread = 2, int nelems = 600) {
   TVirtualStreamerInfo::SetStreamMemberWise(kFALSE);
   TBufferFile buf(TBuffer::kWrite);
   write(buf,1, nelems);
   read(buf,nread);
}
