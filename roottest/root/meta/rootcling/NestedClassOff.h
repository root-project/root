#ifndef _NestedClassOff_h_
#define _NestedClassOff_h_

namespace FdUtil {

  class Fd {
  public:
    union _PixelNumber {
      int fPixelNumber;
      unsigned int fBits;
    };
    typedef _PixelNumber PixelNumberRec;
    typedef _PixelNumber* PixelNumber;
  int i;
  PixelNumberRec j;
  };

};


#endif
