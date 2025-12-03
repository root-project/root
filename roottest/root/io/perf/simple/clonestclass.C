#define var(x) int i##x; float f##x
#define udef(x) i##x(0),f##x(0.0)
#define def(x) i##x(x),f##x(x/3.0)

#include "TNamed.h"
 
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
#include "TClonesArray.h"

void write(TBuffer &buf,int ntimes, int nelems) {
   TClonesArray clones("simple");
   for(int e=0; e<nelems; ++e) {
     new (clones[e]) simple(e);
   }
   buf.SetWriteMode();
   for(int i=0; i<ntimes; ++i) {
      buf.Reset();
      clones.Streamer(buf);
   }
}

void read(TBuffer &buf,int ntimes) {
   TClonesArray clones;
   buf.SetReadMode();
   for(int i=0; i<ntimes; ++i) {
      buf.Reset();
      clones.Streamer(buf);
      // obj->IsA()->Dump(obj);
      // delete obj;
   }
}


void clonestclass(int nread = 2, int nelems = 600) {
   TBufferFile buf(TBuffer::kWrite);
   write(buf,1, nelems);
   read(buf,nread);
}
