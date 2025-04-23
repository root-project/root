//
// Testing transient member setting
//

#include "Riostream.h"
#include "TObjArray.h"

class Unversioned {
public:
   Unversioned(int seed = 4) : fCached(seed) {}

   int fCached; //!
   void Print() {
      cout << "Unversioned::fCached: " << fCached << endl;
   }
};

class ACache {
protected:
   int  x;
   int  y;
   bool zcalc; //! Transient value indicating whether z was cached.
   int  z;     //! Transient value calculated from x and y
   char c;     //  It is x+y
   TObjArray *persist; // Only the objects that need to be saved.
   TObjArray *all;     //! All objects (owner of the objects).
   int  fN;      // Size of fArray
   int *fArray;  //[fN] An array of int that will become an array of char.
   float fValues[3];  // An array of float that will become array of double.
   Unversioned fUnversioned;

 public:
   ACache(int xin = 2, int yin = 3) : x(xin),y(yin),zcalc(false),z(-1),c(x+y),persist(new TObjArray),all(new TObjArray),fN(xin),fArray(0)
   {
      persist->SetName("persist");
      all->SetName("all");
      all->SetOwner(kTRUE);
      fArray = new int[fN];
      for(int i = 0; i < fN; ++i) { fArray[i] = 10+i; }
      for(unsigned int j = 0; j < sizeof(fValues) / sizeof(fValues[0]); ++j) { fValues[j] = j / 100.0; };
   }
   
   ~ACache() {
      delete persist;
      delete all;
      delete [] fArray;
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
      cout << "ACache::fN    " << fN << endl;
      if (fArray) for(int i = 0; i < fN; ++i) { 
         cout << "ACache::fArray["<<i<<"] "<< fArray[i] << endl;
      }
      for(unsigned int j = 0; j < sizeof(fValues) / sizeof(fValues[0]); ++j) { 
         cout << "ACache::fValues["<<j<<"] "<< fValues[j] << endl;

      }
      fUnversioned.Print();
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
#pragma link C++ class Unversioned+;

#pragma read sourceClass="ACache" targetClass="ACache" source="" version="[1-]" target="zcalc" \
   code="{ zcalc = false; }"

#pragma read sourceClass="ACache" version="[1-]" targetClass="ACache" \
    source="TObjArray* persist" target="all" \
    code="{ all->Delete(); all->AddAll(onfile.persist); }"

#pragma read sourceClass="Unversioned" targetClass="Unversioned" source="" version="[1-]" target="fCached" \
    code="{ fCached = 1; }"
#endif

