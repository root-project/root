#ifndef TPROXY_H
#define TPROXY_H

class TTree;
class TBranch;

#include "TTree.h"
#include "TString.h"
#include "TBranchElement.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "Riostream.h"

#include <list>
#include <algorithm>

// Note we could protect the arrays more by introducing a class TArrayWrapper<class T> which somehow knows
// its internal dimensions and check for them ...
// template <class T> TArrayWrapper {
// public:
//    TArrayWrapper(void *where, int dim1);
//    const T operator[](int i) {
//       if (i>=dim1) return 0;
//       return where[i];
//    };
// };
// 2D array would actually be a wrapper of a wrapper i.e. has a method TArrayWrapper<T> operator[](int i);

typedef long sLong64_t; // used to be long long

class TProxy;

void reset(TProxy*);

class TProxyDirector {
public:
   //This class could actually be the selector itself.
   TTree *fTree;
   sLong64_t fEntry;
   std::list<TProxy*> fDirected;

   TProxyDirector(TTree* tree, sLong64_t i) : fTree(tree),fEntry(i) {}
   TProxyDirector(TTree* tree, Int_t i) : fTree(tree),fEntry(i) {} // cint has a problem casting int to long long

   void attach(TProxy* p) {
      fDirected.push_back(p);
   }

   TTree* SetTree(TTree *newtree) {
      TTree* oldtree = fTree;
      fTree = newtree;
      fEntry = -1;
      //if (fInitialized) fInitialized = setup();
      fprintf(stderr,"calling SetTree for %p\n",this);
      std::for_each(fDirected.begin(),fDirected.end(),reset);
      return oldtree;
   }
};

class TProxy {
 public:
   const TString fBranchName; // name of the branch to read
   const char *fDataName;   // name of the element within the branch

   TProxyDirector *fDirector; // contain pointer to TTree and entry to be read
   TBranch *fBranch;           // branch to read
   bool fInitialized;
   sLong64_t fRead;

   TTree *fLastTree;
   void *fWhere;
   Int_t fOffset; // Offset inside the object
   bool fIsaPointer;

   virtual void Print() {
      cout << "fBranchName " << fBranchName << endl;
      cout << "fTree " << fDirector->fTree << endl;
      cout << "fBranch " << fBranch << endl;
   }

   TProxy() : fBranchName(""), fDataName(0), fDirector(0), fBranch(0),
      fInitialized(false), fRead(-1), fLastTree(0), fWhere(0),
      fOffset(0), fIsaPointer(false) {
   };

   TProxy(TProxyDirector* boss, const char* name) : fBranchName(name),
      fDataName(0), fDirector(boss), fBranch(0),
      fInitialized(false), fRead(-1), fLastTree(0), fWhere(0),
      fOffset(0), fIsaPointer(false)
      {
         boss->attach(this);
      }
   TProxy(TProxyDirector* boss, const char* top, const char* name) : 
      fBranchName(top),
      fDataName(0), fDirector(boss), fBranch(0),
      fInitialized(false), fRead(-1), fLastTree(0), fWhere(0),
      fOffset(0), fIsaPointer(false)
      {
         ((TString&)fBranchName).Append(name); 
         boss->attach(this);
      }
   virtual ~TProxy() {};

   void reset() {
      // fprintf(stderr,"Calling reset for %s\n",fBranchName.Data());
      fWhere = 0;
      fBranch = 0;
      fRead = -1;
   }

   bool setup() {
      // Should we check the type?

      if (!fDirector->fTree) {
         return false;
      }
      if (!fBranch) {
         // This does not allow (yet) to precede the branch name with 
         // its mother's name
         fBranch = fDirector->fTree->GetBranch(fBranchName.Data());
         if (!fBranch) return false;


         fWhere = (double*)fBranch->GetAddress();

         if (!fWhere && fBranch->IsA()==TBranchElement::Class()
             && ((TBranchElement*)fBranch)->GetMother()) {
            
            TBranchElement* be = ((TBranchElement*)fBranch);

            be->GetMother()->SetAddress(0);
            fWhere =  (double*)fBranch->GetAddress();

         }
         if (!fWhere) {
            fBranch->SetAddress(0);
            fWhere =  (double*)fBranch->GetAddress();
         }


         if (fWhere && fBranch->IsA()==TBranchElement::Class()) {
            
            TBranchElement* be = ((TBranchElement*)fBranch);

            TStreamerInfo * info = be->GetInfo();
            Int_t id = be->GetID();
            TStreamerElement *elem;
            if (id>=0) {
               fOffset = info->GetOffsets()[id];
               elem = (TStreamerElement*)info->GetElements()->At(id);
               fIsaPointer = elem->IsaPointer();
            }
            
            if (be->GetType()==3 || id<0) {
               // top level TClonesArray or object

               fIsaPointer = false;
               fWhere = be->GetObject();
               
            } else if (be->GetType()==31) {

               fWhere = be->GetObject();

            } else {

               fWhere = ((unsigned char*)fWhere) + fOffset;

            }
         }

      }
      if (fWhere!=0) {
         fLastTree = fDirector->fTree;
         fInitialized = true;
         return true;
      } else {
         return false;
      }
   }

   bool IsInitialized() {
      return (fLastTree == fDirector->fTree) && (fLastTree);
   }
   bool IsaPointer() const {
      return fIsaPointer;
   }

   bool read() { 
      if (fDirector==0) return false;
      //if (fRead<2) fprintf(stderr,"read called %ld %ld\n",fRead,fDirector->fEntry);
      if (fDirector->fEntry!=fRead) {
         if (!IsInitialized()) {
            // fprintf(stderr,"%s proxy not yet initialized\n",fBranchName.Data());
            if (!setup()) {
               fprintf(stderr,"unable to initialize %s\n",fBranchName.Data());
               return false;
            }
         }
         fBranch->GetEntry(fDirector->fEntry);
         fRead = fDirector->fEntry;
      }
      //fprintf(stderr,"at the end of read where is %p\n",fWhere);
      return IsInitialized();
   }


};

void reset(TProxy*x) {x->reset();} 

class TArrayCharProxy : public TProxy {
public:   
   void Print() {
      TProxy::Print();
      cout << "fWhere " << fWhere << endl;
      if (fWhere) cout << "value? " << *(unsigned char*)fWhere << endl;
   }

   TArrayCharProxy() : TProxy() {}
   TArrayCharProxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
   TArrayCharProxy(TProxyDirector *director, const char *top, const char *name) : 
      TProxy(director,top,name) {};
   ~TArrayCharProxy() {};

   unsigned char at(int i) {
      static unsigned char default_val;
      if (!read()) return default_val;
      // should add out-of bound test
      if (IsaPointer()) {
         return (*(( unsigned char**)fWhere))[i];
      } else {
         return (( unsigned char*)fWhere)[i];
      }
   }

   unsigned char operator [](int i) {
      return at(i);
   }
   
   const char* c_str() {
      if (!read()) return "";
      if (IsaPointer()) {
         return *(const char**)fWhere;
      } else {
         return (const char*)fWhere;
      }
   }

   operator std::string() {
      if (!read()) return "";
      if (IsaPointer()) {
         return std::string(*(const char**)fWhere);
      } else {
         return std::string((const char*)fWhere);
      }
   }

};

class TClaProxy : public TProxy {
 public:
   void Print() {
      TProxy::Print();
      cout << "fWhere " << fWhere << endl;
      if (fWhere) {
         if (IsaPointer()) {
            cout << "location " << *(TClonesArray**)fWhere << endl;
         } else {
            cout << "location " << fWhere << endl;
         }
      }
   }

   TClaProxy() : TProxy() {}
   TClaProxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
   TClaProxy(TProxyDirector *director, const char *top, const char *name) : 
      TProxy(director,top,name) {};
   ~TClaProxy() {};

   const TClonesArray* ptr() {
      if (!read()) return 0;
      if (IsaPointer()) {
         return *(TClonesArray**)fWhere;
      } else {
         return (TClonesArray*)fWhere;
      }
   }   

   const TClonesArray* operator->() { return ptr(); }

};

template <class T> 
class TImpProxy : public TProxy {
 public:
   void Print() {
      TProxy::Print();
      cout << "fWhere " << fWhere << endl;
      if (fWhere) cout << "value? " << *(T*)fWhere << endl;
   }

   TImpProxy() : TProxy() {}; 
   TImpProxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
   TImpProxy(TProxyDirector *director, const char *top, const char *name) : 
      TProxy(director,top,name) {};
   ~TImpProxy() {};

   operator T() {
      if (!read()) return 0;
      return *(T*)fWhere;
   }

#ifdef private
#undef private
#define private_was_replaced
#endif
   // For now explicitly disable copying into the value (i.e. the proxy is read-only).
 private:
   TImpProxy(T);
   TImpProxy &operator=(T);
#ifdef private_was_replaced
#define private public
#endif

};

template <class T> 
class TArrayProxy : public TProxy {
 public:
   void Print() {
      TProxy::Print();
      cout << "fWhere " << fWhere << endl;
      if (fWhere) cout << "value? " << *(T*)fWhere << endl;
   }

   TArrayProxy() : TProxy() {}
   TArrayProxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
   TArrayProxy(TProxyDirector *director, const char *top, const char *name) : 
      TProxy(director,top,name) {};
   ~TArrayProxy() {};

   const T& at(int i) {
      static T default_val;
      if (!read()) return default_val;
      // should add out-of bound test
      if (IsaPointer()) {
         return (*((T**)fWhere))[i];
      } else {
         return ((T*)fWhere)[i];
      }
   }

   const T& operator [](int i) {
      return at(i);
   }
   

};

template <class T, int d2, int d3 > 
class TArray3Proxy : public TProxy {
 public:
 typedef T array_t[d2][d3];

   void Print() {
      TProxy::Print();
      cout << "fWhere " << fWhere << endl;
      if (fWhere) cout << "value? " << *(T*)fWhere << endl;
   }

   TArray3Proxy() : TProxy() {}
   TArray3Proxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
   TArray3Proxy(TProxyDirector *director, const char *top, const char *name) : 
      TProxy(director,top,name) {};
   ~TArray3Proxy() {};

   const array_t* at(int i) {
      static array_t default_val;
      if (!read()) return &default_val;
      // should add out-of bound test
      if (IsaPointer()) {
         T *temp = *(T**)fWhere;
         return ((array_t**)temp)[i];
      } else {
         return ((array_t**)fWhere)[i];
      }
   }

   const array_t* operator [](int i) {
      return at(i);
   }

};

template <class T> 
class TClaImpProxy : public TProxy {
 public:

   void Print() {
      TProxy::Print();
      cout << "fWhere " << fWhere << endl;
      if (fWhere) cout << "value? " << *(T*)fWhere << endl;
   }

   TClaImpProxy() : TProxy() {}; 
   TClaImpProxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
   TClaImpProxy(TProxyDirector *director,  const char *top, const char *name) : 
      TProxy(director,top,name) {};
   ~TClaImpProxy() {};

   const T& at(int i) {
      static T default_val;
      if (!read()) return default_val;
      if (fWhere==0) return default_val;
     
      TClonesArray *tca = (TClonesArray*)fWhere;
      
      if (tca->GetLast()<i) return default_val;

      const char *location = (const char*)tca->At(i);
      return *(T*)(location+fOffset);
   }

   const T& operator [](int i) { return at(i); }

#ifdef private
#undef private
#define private_was_replaced
#endif
   // For now explicitly disable copying into the value (i.e. the proxy is read-only).
 private:
   TClaImpProxy(T);
   TClaImpProxy &operator=(T);
#ifdef private_was_replaced
#define private public
#endif

};

template <class T> 
class TClaArrayProxy : public TProxy {
 public:
   typedef T* array_t;

   void Print() {
      TProxy::Print();
      cout << "fWhere " << fWhere << endl;
      if (fWhere) cout << "value? " << *(T*)fWhere << endl;
   }

   TClaArrayProxy() : TProxy() {}
   TClaArrayProxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
   TClaArrayProxy(TProxyDirector *director, const char *top, const char *name) : 
      TProxy(director,top,name) {};
   ~TClaArrayProxy() {};

   const array_t at(int i) {
      static T default_val;
      if (!read()) return &default_val;
      if (fWhere==0) return &default_val;
     
      TClonesArray *tca = (TClonesArray*)fWhere;
      
      if (tca->GetLast()<i) return &default_val;

      const char *location = (const char*)tca->At(i);
      location += fOffset;

      if (IsaPointer()) {
         return (array_t)( *(T**)location );
      } else {
         return (array_t)location;
      }
   }

   const array_t operator [](int i) { return at(i); }

};

template <class T, int d2 > 
class TClaArray2Proxy : public TProxy {
 public:
 typedef T array_t[d2];

   void Print() {
      TProxy::Print();
      cout << "fWhere " << fWhere << endl;
      if (fWhere) cout << "value? " << *(T*)fWhere << endl;
   }

   TClaArray2Proxy() : TProxy() {}
   TClaArray2Proxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
   TClaArray2Proxy(TProxyDirector *director, const char *top, const char *name) : 
      TProxy(director,top,name) {};
   ~TClaArray2Proxy() {};

   const array_t &at(int i) {
      static array_t default_val;
      if (!read()) return &default_val;
      if (fWhere==0) return &default_val;
     
      TClonesArray *tca = (TClonesArray*)fWhere;
      
      if (tca->GetLast()<i) return &default_val;

      const char *location = (const char*)tca->At(i);
      location += fOffset;

      if (IsaPointer()) {
         T *temp = *(T**)location;
         return ((array_t**)temp)[i];
      } else {
         return ((array_t**)location)[i];
      }
   }

   const array_t &operator [](int i) { return at(i); }

};

template <class T, int d2, int d3 > 
class TClaArray3Proxy : public TProxy {
 public:
 typedef T array_t[d2][d3];

   void Print() {
      TProxy::Print();
      cout << "fWhere " << fWhere << endl;
      if (fWhere) cout << "value? " << *(T*)fWhere << endl;
   }

   TClaArray3Proxy() : TProxy() {}
   TClaArray3Proxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
   TClaArray3Proxy(TProxyDirector *director, const char *top, const char *name) : 
      TProxy(director,top,name) {};
   ~TClaArray3Proxy() {};

   const array_t* at(int i) {
      static array_t default_val;
      if (!read()) return &default_val;
      if (fWhere==0) return &default_val;
     
      TClonesArray *tca = (TClonesArray*)fWhere;
      
      if (tca->GetLast()<i) return &default_val;

      const char *location = (const char*)tca->At(i);
      location += fOffset;

      if (IsaPointer()) {
         T *temp = *(T**)location;
         return ((array_t**)temp)[i];
      } else {
         return ((array_t**)location)[i];
      }
   }

   const array_t* operator [](int i) { return at(i); }

};

//TImpProxy<TObject> d;
typedef TImpProxy<double>   TDoubleProxy;
typedef TImpProxy<float>    TFloatProxy;
typedef TImpProxy<UInt_t>   TUIntProxy;
typedef TImpProxy<UShort_t> TUShortProxy;
typedef TImpProxy<UChar_t>  TUCharProxy;
typedef TImpProxy<Int_t>    TIntProxy;
typedef TImpProxy<Short_t>  TShortProxy;
typedef TImpProxy<Char_t>   TCharProxy;

typedef TArrayProxy<double>   TArrayDoubleProxy;
typedef TArrayProxy<float>    TArrayFloatProxy;
typedef TArrayProxy<UInt_t>   TArrayUIntProxy;
typedef TArrayProxy<UShort_t> TArrayUShortProxy;
typedef TArrayProxy<UChar_t>  TArrayUCharProxy;
typedef TArrayProxy<Int_t>    TArrayIntProxy;
typedef TArrayProxy<UShort_t> TArrayShortProxy;
//specialized ! typedef TArrayProxy<Char_t>  TArrayCharProxy;

typedef TClaImpProxy<double>   TClaDoubleProxy;
typedef TClaImpProxy<float>    TClaFloatProxy;
typedef TClaImpProxy<UInt_t>   TClaUIntProxy;
typedef TClaImpProxy<UShort_t> TClaUShortProxy;
typedef TClaImpProxy<UChar_t>  TClaUCharProxy;
typedef TClaImpProxy<Int_t>    TClaIntProxy;
typedef TClaImpProxy<Short_t>  TClaShortProxy;
typedef TClaImpProxy<Char_t>   TClaCharProxy;

#endif // TPROXY_H


