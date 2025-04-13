//
// Testing transient member setting
//

#include "Riostream.h"

// This class used to be named ACache
class Axis {
protected:
   int   z; // It is 1000*x + 10*y
   float c; // It is x+y
   int   fN; // Size of fArray
   char *fArray; //[fN] Array that used to be an array of int
   double fValues[3];  // An array of double that used to be array of float.

 public:
   Axis() : z(-1),c(-1),fN(0),fArray(0) {}
   ~Axis() { delete [] fArray; }

   int GetX() { return z/1000; }
   int GetY() { return (z%1000)/10; }

   int GetZ() {
      return z;
   }

   void Print() {
      cout << "Axis::x " << GetX() << endl;
      cout << "Axis::y " << GetY() << endl;
      cout << "Axis::z " << GetZ() << endl;
      cout << "Axis::c " << c << endl;
      cout << "Axis::fN    " << fN << endl;
      if (fArray) for(int i = 0; i < fN; ++i) { 
         cout << "Axis::fArray["<<i<<"] "<< (Short_t)fArray[i] << endl;
      }
      for(unsigned int j = 0; j < sizeof(fValues) / sizeof(fValues[0]); ++j) { 
         cout << "ACache::fValues["<<j<<"] "<< fValues[j] << endl;
         
      }
   }
};

class Container {
public:
   Axis a;
};


#ifdef __MAKECINT__
#pragma link C++  options=version(2) class Axis+;
#pragma read sourceClass="ACache" targetClass="Axis" source="int x; int y;"  version="[8]" target="z" \
   code="{ z = onfile.x*1000 + onfile.y*10; }"
#pragma read sourceClass="ACache" version="[9]" targetClass="Axis";
#pragma link C++ options=version(3) class Container+;
#endif

