#if !defined(__CINT__) || defined(__MAKECINT__)
#include "TProxy.h"
#endif

#define InjectProxyInterface()                           \
   TProxy *proxy() { return obj.proxy(); }               \
   void reset() { obj.reset(); }                         \
   bool setup() { return obj.setup(); }                  \
   bool IsInitialized() { return obj.IsInitialized(); }  \
   bool IsaPointer() const { return obj.IsaPointer(); }  \
   bool read() { return obj.read(); } 

template <class T> 
class TObjProxy {
  TProxy obj;
public:
  InjectProxyInterface();

  TObjProxy() : obj() {}; 
  TObjProxy(TProxyDirector *director, const char *name) : obj(director,name) {};
  TObjProxy(TProxyDirector *director, const char *top, const char *name) : 
      obj(director,top,name) {};
  TObjProxy(TProxyDirector *director, const char *top, const char *name, const char *data) : 
      obj(director,top,name,data) {};
  TObjProxy(TProxyDirector *director, TProxy *parent, const char *name) : obj(director,parent, name) {};
  ~TObjProxy() {};

   void Print() {
      obj.Print();
      cout << "fWhere " << obj.fWhere << endl;
      if (obj.fWhere) cout << "address? " << (T*)obj.fWhere << endl;
   }

   T* ptr() {
      //static T default_val;
      if (!obj.read()) return 0; // &default_val;
      T *temp = (T*)obj.GetStart();
      // if (temp==0) return &default_val;
      return temp;
   }
 
   T* operator->() { return ptr(); }
   operator T*() { return ptr(); }
   // operator T&() { return *ptr(); }

};

template <class T, int d2 >
class TArray2Proxy {
 public:
   TProxy obj;
   InjectProxyInterface();
   
   TArray2Proxy() : obj() {}
   TArray2Proxy(TProxyDirector *director, const char *name) : obj(director,name) {};
   TArray2Proxy(TProxyDirector *director, const char *top, const char *name) : 
      obj(director,top,name) {};
   TArray2Proxy(TProxyDirector *director, const char *top, const char *name, const char *data) : 
      obj(director,top,name,data) {};
   TArray2Proxy(TProxyDirector *director, TProxy *parent, const char *name) : obj(director, parent, name) {};
   ~TArray2Proxy() {};

   typedef T array_t[d2];

   void Print() {
      TProxy::Print();
      cout << "fWhere " << obj.fWhere << endl;
      if (obj.fWhere) cout << "value? " << *(T*)obj.fWhere << endl;
   }

   const array_t &at(int i) {
      static array_t default_val;
      if (!obj.read()) return default_val;
      // should add out-of bound test
      array_t *arr = 0;
      arr = (array_t*)((T*)(obj.GetStart()));
      if (arr) return arr[i];
      else return default_val;
   }

   const array_t &operator [](int i) { return at(i); }

};

template <class T> 
class TClaObjProxy  {
   TClaProxy obj;
 public:
   InjectProxyInterface();

   void Print() {
      obj.Print();
      cout << "obj.fWhere " << obj.fWhere << endl;
      //if (obj.fWhere) cout << "value? " << *(T*)obj.fWhere << endl;
   }

   TClaObjProxy() : obj() {}; 
   TClaObjProxy(TProxyDirector *director, const char *name) : obj(director,name) {};
   TClaObjProxy(TProxyDirector *director,  const char *top, const char *name) : 
      obj(director,top,name) {};
   TClaObjProxy(TProxyDirector *director,  const char *top, const char *name, const char *data) : 
      obj(director,top,name,data) {};
   TClaObjProxy(TProxyDirector *director, TProxy *parent, const char *name) : obj(director,parent, name) {};
   ~TClaObjProxy() {};

   const T* at(int i) {
      static T default_val;
      if (!obj.read()) return &default_val;
      if (obj.fWhere==0) return &default_val;

      T* temp = (T*)obj.GetClaStart(i);
      if (temp) return temp;
      else return &default_val;
   }

   const T* operator [](int i) { return at(i); }

};
