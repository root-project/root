// FixedArray.C
//
// A set of classes for testing schema evolution
// where a fixed size array changes size.
//

#include "TObject.h"
#include <iostream>

using namespace std;

class FixedArray : public TObject {
public:
#if (VERSION == 1)
  enum { SIZE = 3 };
#else
  enum { SIZE = 5 };
#endif
private:
  int fDummy1;
  int fDummy2;
  int arr[SIZE];
public:
  FixedArray();
  ~FixedArray() override;
  void check();
  void set();
#if (VERSION==1)
  ClassDefOverride(FixedArray, 1);
#else
  ClassDefOverride(FixedArray, 2);
#endif
};

#ifdef __ROOTCLING__
#pragma link C++ class FixedArray+;
#endif

FixedArray::FixedArray()
{
   fDummy1 = -1;
   fDummy2 = -1;
   for (int i = 0; i < SIZE; i++) {
      arr[i] = -1;
   }
}

FixedArray::~FixedArray()
{
}

void FixedArray::check()
{
   cerr << "SIZE: " << SIZE << endl;
   for (int i = 0; i < SIZE; i++) {
      cerr << i << ": " << arr[i] << endl;
   }
}

void FixedArray::set()
{
   for (int i = 0; i < SIZE; i++) {
      arr[i] = i * 10;
   }
}

class Level2 {
private:
   FixedArray fArray;
public:
   Level2();
   virtual ~Level2();
   ClassDef(Level2, 2);
};

#ifdef __ROOTCLING__
#pragma link C++ class Level2+;
#endif

Level2::Level2()
{
}

Level2::~Level2()
{
}

class FixedArrayContainer : public TObject, public Level2 {
private:
   //Level2 fLevel2;
public:
   FixedArrayContainer();
   ~FixedArrayContainer() override;
   ClassDefOverride(FixedArrayContainer, 2);
};

#ifdef __ROOTCLING__
#pragma link C++ class FixedArrayContainer+;
#endif

FixedArrayContainer::FixedArrayContainer()
{
}

FixedArrayContainer::~FixedArrayContainer()
{
}

