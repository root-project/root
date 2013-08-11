#include <vector>
#include "TObject.h"

class MoreNested {
public:
   MoreNested(int i = 0) : fMoreNestedValue(i) {}
   int fMoreNestedValue;
   virtual ~MoreNested() {}
   ClassDef(MoreNested,2);
};

class Nested {
 public:
    Nested(int i = 0) : fNestedValue(i),fOther(i) {}
   int fNestedValue;
   MoreNested fOther;
   virtual ~Nested() {}
   ClassDef(Nested,2);
};

class Inside {
 public:
   Inside(int i = 0) : fValue(i),fSub(i) {}
   int fValue;
   Nested fSub;
   virtual ~Inside() {}
   ClassDef(Inside,2);
};

class colClass {
public:
   void Fill(int howmany) {
      for(int i = 0; i < howmany; ++i) {
         fValues.push_back(i);
         fObjects.push_back(Inside(i));
         fMeans.push_back(i/2.0);
      }
   }

   std::vector<int>    fValues;
   std::vector<Inside> fObjects;
   std::vector<double> fMeans;

   virtual ~colClass() {}
   ClassDef(colClass,2);
};

#ifdef __MAKECINT__
#pragma link C++ class MoreNested+;
#pragma link C++ class Nested+;
#pragma link C++ class Inside+;
#pragma link C++ class colClass+;
#endif
