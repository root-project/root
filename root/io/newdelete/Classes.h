#include "TObject.h"
#include "TSocket.h"

bool Classes();

//void *operator new(int); 

class PrivateDestructor {
  friend TObject;
  ~PrivateDestructor();
public:
  int a;
  PrivateDestructor(int b) : a(b) {};
};

class Struct {
public:
  int a;
};

class Nodefault {
public:
  int a;
  Nodefault(int aa) : a(aa) {};
  ~Nodefault() { fprintf(stderr,"deleting a Nodefault %d\n",a); }
};

class NodefaultOpNew {
public:
  int a;
  NodefaultOpNew(int aa) : a(aa) {};
  void *operator new(size_t sz) { return TStorage::ObjectAlloc(sz); }
};

class Default {
public:
  int a;
  Default(int aa = 0) : a(aa) {};
};

class PartialDefault {
public:
  TClass *c;
  int a;
  PartialDefault(TClass* cl,Int_t all=0): c(cl),a(all) {};
};

class Normal {
public:
  int a;
  Normal() : a(0) {};
  Normal(int aa) : a(aa) {};
};


class TStruct : public TObject {
public:
  int a;
  ClassDef(TStruct,0);
};

class TNodefault : public TObject {
public:
  int a;
  TNodefault(int aa) : a(aa) {};
  virtual ~TNodefault() { fprintf(stderr,"deleting a TNodefault %d\n",a); }
  ClassDef(TNodefault,1); //
};

class TNodefaultOpNew : public TObject  {
public:
  int a;
  TNodefaultOpNew(int aa) : a(aa) {};
  void *operator new(size_t sz) { return TStorage::ObjectAlloc(sz); };
  ClassDef(TNodefaultOpNew,0);
};

class TDefault  : public TObject{
public:
  int a;
  TDefault(int aa = 0) : a(aa) {};
  ClassDef(TDefault,0);
};

class TPartialDefault  : public TObject  {
public:
  TClass *c;
  int a;
  TPartialDefault(TClass* cl,Int_t all=0) : c(cl),a(all) {}
  ClassDef(TPartialDefault,0);
};

class TNormal  : public TObject{
public:
  int a;
  TSocket *s;
  TNormal() : a(0) {};
  TNormal(int aa) : a(aa) {};

  ClassDef(TNormal,1);
};

