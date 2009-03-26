//
// Testing transient member setting
//

#include "Riostream.h"
#include "TObjArray.h"

class ACache {
protected:
   int  x;
   int  y;
   bool zcalc; //! Transient value indicating whether z was cached.
   int  z;     //! Transient value calculated from x and y
   char c;     //  It is x+y
   TObjArray *persist; // Only the objects that need to be saved.
   TObjArray *all;     //! All objects (owner of the objects).

 public:
   ACache(int xin = 2, int yin = 3) : x(xin),y(yin),zcalc(false),z(-1),c(x+y),persist(new TObjArray),all(new TObjArray)
   {
      persist->SetName("persist");
      all->SetName("all");
      all->SetOwner(kTRUE);
   }
   
   ~ACache() {
      delete persist;
      delete all;
   }
   
   void CreateObjs()
   {
      // Populate the array.
      
      TNamed * n = new TNamed("objectone","for file");
      persist->Add(n);
      all->Add(n);
      n = new TNamed("objecttwo","for memory");
      all->Add(n);
   }
   
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
      persist->Print();
      all->Print();
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

#pragma read sourceClass="ACache" version="[1-]" targetClass="ACache" \
    source="TObjArray* persist" target="all" \
    code="{ all->Delete(); all->AddAll(onfile.persist); }"

#endif

