#ifndef RootPCellID_h
#define RootPCellID_h

#include <string>
#include <iostream>
#include <vector>
using std::vector;
// #include "TObject.h"

class TBuffer;

class crap {};

class RootPCellID : public crap {

public:

  RootPCellID(){    
    for(int j=0;j<4;j++) base[j]=0;
  }
  
  RootPCellID(const std::string & b, unsigned int i):id(i) {
    for(int j=0;j<4;j++) base[j]=b[j];
  }
  virtual void Print() {
    std::cout << "base \t";
    for(int j=0;j<4;j++) std::cout << base[j] << " ";
    std::cout  << std::endl<< "id \t" << id << std::endl;

  }
#if 0
  virtual void Streamer(TBuffer &R__b) {
   // Stream an object of class RootPCellID.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      R__b >> id;
      R__b.ReadStaticArray(base);
   } else {
      R__b << id;
      R__b.WriteArray(base, 4);
   }
  }
#endif

private:
  unsigned int    id;
  char base[4];
  // ClassDef(RootPCellID,1)
};

class RootPCfix : public RootPCellID {
public:
   RootPCfix() : RootPCellID("none",0),fix(0) {}
   RootPCfix(int i) : RootPCellID("fix1",i),fix(33) {}
   int fix;
   void Print() {
     RootPCellID::Print();
     std::cout  << "fix \t" << fix << std::endl;
   }
};

class RootPCvirt : public RootPCellID {
public:
   RootPCvirt() : RootPCellID("none",0),virt(0) {}
   RootPCvirt(int v) :  RootPCellID("virt",v),virt(44) {}
   int virt;
   vector<int> list;
   vector<RootPCfix*> list2;
   virtual ~RootPCvirt() {};
   void Print() {
     RootPCellID::Print();
     std::cout  << "virt \t" << virt << std::endl;
   }
   virtual int getvirt() { return virt; }
};

class RootPCnodict : public RootPCellID {
public:
   RootPCnodict() : RootPCellID("none",0),nodict(0) {}
   RootPCnodict(int n) :  RootPCellID("nodict",n),nodict(55) {}
   int nodict;
   void Print() {
     RootPCellID::Print();
     std::cout  << "nodict \t" << nodict << std::endl;
   }
};

template <class T> class RootPCtemp : public RootPCellID {
public:
   RootPCtemp() : RootPCellID("none",0),temp(0) {}
   RootPCtemp(T n) :  RootPCellID("template",n),temp(66) {}
   T temp;
   void Print() {
     RootPCellID::Print();
     std::cout  << "templated \t" << temp << std::endl;
   }
};

//inline Short_t GetClassVersion(RootPCellID*) { return 2; }
// or template<> inline Short_t GetClassVersion<RootPCellID >(RootPCellID*) { return 2; }
// or template<> inline Short_t GetClassVersion<RootPCellID >() { return 2; }

// plus maybe for class template.
//  inline template <class T> Short_t GetClassVersion<RootPCtemp<T> >(RootPCtemp<T> *) { return 4; }

#endif






