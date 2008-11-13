//
// Testing transient member setting
//

#include "Riostream.h"

class ACache {
protected:
   int  x;
   int  y;
   bool zcalc; //! Transient value indicating whether z was cached.
   int  z;     //! Transient value calculated from x and y
   char c;     //  It is x+y

 public:
   ACache(int xin = 2, int yin = 3) : x(xin),y(yin),zcalc(false),z(-1),c(x+y) {}

   int GetX() { return x; }
   int GetY() { return y; }

   int GetZ() {
      if (zcalc) return z;
      z = x*1000+y*10;
      zcalc = true;
      return z;
   }

   void Print() {
      cout << "ACache::x     " << x << endl;
      cout << "ACache::y:    " << y << endl;
      cout << "ACache::zcalc " << zcalc << endl;
      cout << "ACache::z     " << z << endl;
      cout << "ACache::c     " << (short)c << endl;
   }
};

class Container {
public:
   Container(int seed = 4) : a(seed+1,seed+2) {}

   ACache a;
};

#ifdef __MAKECINT__
#pragma link C++ options=version(8) class ACache+;
#pragma link C++ options=version(2) class Container+;
#pragma read sourceClass="ACache" targetClass="ACache" source="" version="[1-]" target="zcalc" \
   code="{ zcalc = false; }"
#endif

