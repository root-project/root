#include "TObject.h"
#include "TSocket.h"

bool Classes();

//void *operator new(int);

class PrivateDestructor {
  friend class TObject;
  ~PrivateDestructor();
public:
  int a;
  PrivateDestructor(int b) : a(b) {}
};

class Struct {
public:
  int a;
};

class Nodefault {
public:
  int a;
  Nodefault(int aa) : a(aa) {}
  ~Nodefault() { fprintf(stderr,"deleting a Nodefault %d\n",a); }
};

class NodefaultOpNew {
public:
  int a;
  NodefaultOpNew(int aa) : a(aa) {}
  void *operator new(size_t sz) { return TStorage::ObjectAlloc(sz); }
};

class OpNew {
public:
  int a;
  OpNew() : a(0) {}
  void *operator new(size_t sz) { return TStorage::ObjectAlloc(sz); }
};

class OpNewPlacement {
public:
  int a;
  OpNewPlacement() : a(0) {}
  void *operator new(size_t sz, void *p) { return TStorage::ObjectAlloc(sz,p); }
  void     operator delete(void *ptr, void *vp);
};

class OpNewBoth {
public:
  int a;
  OpNewBoth() : a(0) {}
  void *operator new(size_t sz) { return TStorage::ObjectAlloc(sz); }
  void *operator new(size_t sz, void *p) { return TStorage::ObjectAlloc(sz,p); }
/*   void     operator delete(void *ptr, void *vp); */
/*   void     operator delete[](void *ptr, void *vp); */
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
  PartialDefault(TClass* cl,Int_t all=0): c(cl),a(all) {}
};

class Normal {
public:
  int a;
  Normal() : a(0) {}
  Normal(int aa) : a(aa) {}
};


class TStruct : public TObject {
public:
  int a;
  ClassDefOverride(TStruct,0);
};

class TNodefault : public TObject {
public:
  int a;
  TNodefault(int aa) : a(aa) {}
  ~TNodefault() override { fprintf(stderr,"deleting a TNodefault %d\n",a); }
  ClassDefOverride(TNodefault,1); //
};

class TNodefaultOpNew : public TObject  {
public:
  int a;
  TNodefaultOpNew(int aa) : a(aa) {}
  void *operator new(size_t sz) { return TStorage::ObjectAlloc(sz); }
  ClassDefOverride(TNodefaultOpNew,0);
};

class TOpNew : public TObject  {
public:
  int a;
  TOpNew() : a(0) {}
  void *operator new(size_t sz) { return TStorage::ObjectAlloc(sz); }
  ClassDefOverride(TOpNew,1);
};

class TOpNewPlacement : public TObject {
public:
  int a;
  TOpNewPlacement() : a(0) {}
  void *operator new(size_t sz, void *p) { return TStorage::ObjectAlloc(sz,p); }
  ClassDefOverride(TOpNewPlacement,1);
};

class TOpNewBoth : public TObject {
public:
  int a;
  TOpNewBoth() : a(0) {}
  void *operator new(size_t sz) { return TStorage::ObjectAlloc(sz); }
  void *operator new(size_t sz, void *p) { return TStorage::ObjectAlloc(sz,p); }
  ClassDefOverride(TOpNewBoth,1);
};

class TDefault  : public TObject{
public:
  int a;
  TDefault(int aa = 0) : a(aa) {}
  ClassDefOverride(TDefault,0);
};

class TPartialDefault  : public TObject  {
public:
  TClass *c;
  int a;
  TPartialDefault(TClass* cl,Int_t all=0) : c(cl),a(all) {}
  ClassDefOverride(TPartialDefault,0);
};

class TPrivateDefaultConstructor : public TObject {
private:
   TPrivateDefaultConstructor() {}
public:
   TPrivateDefaultConstructor(int) {}
   ClassDefOverride(TPrivateDefaultConstructor,1);
};

class TNormal  : public TObject{
public:
  int a;
  TPrivateDefaultConstructor *s;
  TNormal() : a(0) {}
  TNormal(int aa) : a(aa) {}

  ClassDefOverride(TNormal,1);
};

// See ROOT-7014.
struct ROOT7014_class {
   template<class... ARGS>
   ROOT7014_class(ARGS...) {}
   ROOT7014_class(int CannotUseThis);
};
