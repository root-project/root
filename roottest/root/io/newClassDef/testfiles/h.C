// Checking that TArrayI do not work .....


#include "TArrayI.h"
#include "TBuffer.h"

void write(TBuffer &buf) {

  TArrayI * array = new TArrayI(20);
  for(int i=0; i<10; i++) {

    (*array)[i] = i;

  }
  buf.Reset();
  buf.SetWriteMode();

  buf << array;

};

void read(TBuffer &buf) {

  buf.Reset();
  buf.SetReadMode();

  TArrayI * array; // = new TArrayI();
  
  buf >> array;
  for(int i=0; i<10; i++) {

    fprintf(stderr,"%d : %d \n", i, (*array)[i]);

  }

};

void driver() {
  TBuffer* buf = new TBuffer(TBuffer::kWrite);
  write(*buf);
  read(*buf);
}
