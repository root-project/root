#include "InheritMulti.h"
#include "TBufferFile.h"

bool write(TBuffer &buf) {

  MyInverseMulti * im = new MyInverseMulti(1,2);
  MyMulti * m = new MyMulti(3,4);

  buf.Reset();
  buf.SetWriteMode();

  fprintf(stderr,"Will write the objects\n");
  buf << im;
  buf << m;

  return true;
};

bool read(TBuffer &buf) {

  buf.Reset();
  buf.SetReadMode();

  MyInverseMulti * im = 0;
  MyMulti * m = 0;

  fprintf(stderr,"Will read the objects\n");
  buf >> im;
  buf >> m;
  
  //fprintf(stderr,"im is worth %p\n",im);
  //fprintf(stderr,"im->t is worth %d\n",im->t);


  bool result = (im!=0 && im->t==1 && im->i==2);
  if (!result) { if (im==0) fprintf(stderr,"im is null\n"); else im->Dump(); }
  result &= (m!=0 && m->t==3 && m->m==4);
  if (!result) { if (m==0) fprintf(stderr,"m is null\n"); m->Dump(); }

  
  return result;

};


bool InheritMulti_driver() {
  TBuffer* buf = new TBufferFile(TBuffer::kWrite);
  bool result = write(*buf);
  result &= read(*buf);
   
  return result;
}
