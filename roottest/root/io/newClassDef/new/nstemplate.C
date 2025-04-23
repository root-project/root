#include "nstemplate.h"
#include "TBufferFile.h"

const double dvalue = 33.3;

ClassImp(MySpace::MyTemplate<const double*>)

templateClassImp(MySpace::MyTemplate)

   //problem should be in namespace
  ClassImp(MySpace::MyTemplate<int>)

namespace MySpace {


  static MyTemplate<int> dummy(1); 
  static MyTemplate <const double*> dummy4(&dvalue);

  static MyTemplate< int> *pdummy; 
  static MyTemplate <const double*> *pdummy4;


  
}

TBuffer* nt_writetest() 
{
  TBuffer *b = new TBufferFile(TBuffer::kWrite);
  *b << &MySpace::dummy;
  *b << &MySpace::dummy4;
  return b;

};

void nt_readtest(TBuffer & b) 
{
  // TBuffer b(TBuffer::kRead);
  b >> MySpace::pdummy;
  if (MySpace::pdummy->variable!=MySpace::dummy.variable) {
     fprintf(stderr,"Error: MySpace::MyTemplate<int> not read properly!");
     fprintf(stderr,"Expected %d and got %d\n", 
             MySpace::dummy.variable,
             MySpace::pdummy->variable);
  }
  b >>  MySpace::pdummy4;
  if ( MySpace::pdummy4->variable!= MySpace::dummy4.variable) {
     fprintf(stderr,"Error: MyTemplate<const double*> not read properly!");
     fprintf(stderr,"Expected %f and got %f\n", 
             MySpace::dummy4.variable,
             MySpace::pdummy4->variable);
  }

}
void nstemplate_driver() {
  TBuffer *buf = nt_writetest();
  buf->SetReadMode();
  buf->Reset();
  nt_readtest(*buf);
}
