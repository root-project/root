#ifndef RootPCellID_h
#define RootPCellID_h

#include <string>
#include <iostream>
#include <vector>
using std::vector;
#include <string>
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
  virtual ~RootPCellID() {};

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

template <class T> class helper {};

class RootPCvirt : public RootPCellID {
public:
   RootPCvirt() : RootPCellID("none",0),virt(0) {}
   RootPCvirt(int v) :  RootPCellID("virt",v),virt(44) {}
   int virt;
   vector<int> list;
   vector<RootPCfix*> list2;
   vector<helper<float>* > list3; //!
   vector<helper<float*> > list4; //!
   vector<vector<float*> > list5; //!
   vector<vector<float>* > list6; //!
   typedef double value;
   value vv;
   std::string s;
   enum Enumeration { kInit, kSecond };
   Enumeration status;
#ifdef INC_FAILURE
   union Stuff { int a; double b; };
   Stuff stuff;
#endif
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
   vector<RootPCtemp<T>*> list;//!
   void Print() {
     RootPCellID::Print();
     std::cout  << "templated \t" << temp << std::endl;
   }
};

template <class T> class RootPCtempObj : public RootPCellID {
public:
   RootPCtempObj() : RootPCellID("none",0),temp(0),temp2(0)  {}
   RootPCtempObj(T n) :  RootPCellID("template",-11),temp(0),temp2(0) {}
   RootPCtempObj(const RootPCtempObj &) : RootPCellID("none",0),temp(),temp2()  {}

   T *temp; //!
   typedef T *value;
   value temp2; //!
   vector<RootPCtempObj<T>*> list;//!
   void Print() {
     RootPCellID::Print();
     //std::cout  << "templated \t" << temp << std::endl;
   }
};

namespace Local {

   class RootPCtop : public RootPCellID {
   public:
      RootPCtop() {};
   };

   class RootPCbottom : public RootPCtop {
   public:
      RootPCbottom() {};
      RootPCtop var;
   };

}


class RootPCobject : public RootPCellID, public TObject {
public:
   RootPCobject() : RootPCellID("none",0),obj(0) {}
   RootPCobject(int n) :  RootPCellID("obj1",n),obj(101) {}
   virtual ~RootPCobject() {};
   int obj;
   void Print() {
     RootPCellID::Print();
     std::cout  << "obj \t" << obj << std::endl;
     //Dump();
   }   
   ClassDef(RootPCobject,1) // inherit second from TObject
};

class RootPCobject2 : public TObject, public RootPCellID {
public:
   RootPCobject2() : RootPCellID("none",0),obj(0) {}
   RootPCobject2(int n) :  RootPCellID("obj2",n),obj(102) {}
   virtual ~RootPCobject2() {};
   int obj;
   void Print() {
     RootPCellID::Print();
     std::cout  << "obj \t" << obj << std::endl;
     //Dump();
   }   
   ClassDef(RootPCobject2,1) // inherit first from TObject
};

class RootPCmisClDef : public RootPCellID, public TObject  {
 public:
   RootPCmisClDef() : RootPCellID("none",0),obj(0) {}
   RootPCmisClDef(int n) :  RootPCellID("miss",n),obj(103) {}
   virtual ~RootPCmisClDef() {};
   int obj;
   void Print() {
     RootPCellID::Print();
     std::cout  << "obj \t" << obj << std::endl;
     //Dump();
   }   
   // intentionally NOT putting the ClasDef

};

class RootPrivPCobject : RootPCellID, public TObject {
public:
   RootPrivPCobject() : RootPCellID("none",0),obj(0) {}
   RootPrivPCobject(int n) :  RootPCellID("obj1",n),obj(101) {}
   virtual ~RootPrivPCobject() {};
   int obj;
   void Print() {
     RootPCellID::Print();
     std::cout  << "obj \t" << obj << std::endl;
     //Dump();
   }   
   ClassDef(RootPrivPCobject,1) // inherit second from TObject
};

class RootPrivPCobject2 :  public TObject, private RootPCellID {
public:
   RootPrivPCobject2() : RootPCellID("none",0),obj(0) {}
   RootPrivPCobject2(int n) :  RootPCellID("obj1",n),obj(101) {}
   virtual ~RootPrivPCobject2() {};
   int obj;
   void Print() {
     RootPCellID::Print();
     std::cout  << "obj \t" << obj << std::endl;
     //Dump();
   }   
   ClassDef(RootPrivPCobject2,1) // inherit second from TObject
};

class RootPrivPC : RootPCellID {
public:
   RootPrivPC() : RootPCellID("none",0),obj(0) {}
   RootPrivPC(int n) :  RootPCellID("obj1",n),obj(101) {}
   virtual ~RootPrivPC() {};
   int obj;
   void Print() {
     RootPCellID::Print();
     std::cout  << "obj \t" << obj << std::endl;
     //Dump();
   }   
};

//inline Short_t GetClassVersion(RootPCellID*) { return 2; }
// or template<> inline Short_t GetClassVersion<RootPCellID >(RootPCellID*) { return 2; }
// or template<> inline Short_t GetClassVersion<RootPCellID >() { return 2; }

// plus maybe for class template.
//  inline template <class T> Short_t GetClassVersion<RootPCtemp<T> >(RootPCtemp<T> *) { return 4; }

#endif






