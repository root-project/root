#include "InheritMulti.h"
#include "TBufferFile.h"

void A::Streamer(TBuffer &R__b)
{
   // Stream an object of class A.
   Int_t dummy = 7;

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      dummy = 0;

      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
      TObject::Streamer(R__b);
      R__b >> dummy;
      if (dummy != 7) fprintf(stderr,"Error in A::Streamer, the dummy variable is: %d\n",dummy);
      R__b >> a;
      R__b.CheckByteCount(R__s, R__c, A::IsA());
   } else {
      R__c = R__b.WriteVersion(A::IsA(), kTRUE);
      TObject::Streamer(R__b);
      R__b << dummy;
      R__b << a;
      R__b.SetByteCount(R__c, kTRUE);
   }
}

bool write(TBuffer &buf) {

  MyInverseMulti * im = new MyInverseMulti(1,2);
  MyMulti * m = new MyMulti(3,4);
  B * b = new B(5,6);

  buf.Reset();
  buf.SetWriteMode();

  fprintf(stderr,"Will write the objects\n");
  buf << im;
  buf << m;
  b->Streamer(buf);

  return true;
};

bool read(TBuffer &buf) {

  buf.Reset();
  buf.SetReadMode();

  MyInverseMulti * im = 0;
  MyMulti * m = 0;
  B * b = new B;

  fprintf(stderr,"Will read the objects\n");
  buf >> im;
  buf >> m;
  fprintf(stderr,"Will read the object b\n");
  b->Streamer(buf);
  
  //fprintf(stderr,"im is worth %p\n",im);
  //fprintf(stderr,"im->t is worth %d\n",im->t);


  bool result = (im!=0 && im->t==1 && im->i==2);
  if (!result) { if (im==0) fprintf(stderr,"im is null\n"); else im->Dump(); }
  result &= (m!=0 && m->t==3 && m->m==4);
  if (!result) { if (m==0) fprintf(stderr,"m is null\n"); m->Dump(); }
  fprintf(stderr,"Will test the object b\n");
  result &= (b!=0 && b->b==6 && b->a!=0 && b->a->a==5);
  if (!result) { if (b==0) fprintf(stderr,"b is null\n"); b->Dump(); }
  
  return result;

};


bool InheritMulti_driver() {
  TBuffer* buf = new TBufferFile(TBuffer::kWrite);
  bool result = write(*buf);
  result &= read(*buf);
   
  return result;
}
