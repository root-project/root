#ifndef TPROXY_H
#define TPROXY_H

#include "TProxyDirector.h"
using namespace ROOT;

class TTree;
class TBranch;

#include "TTree.h"
#include "TString.h"
#include "TBranchElement.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "Riostream.h"
#include "TError.h"

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

namespace ROOT {

   class TProxyHelper {
   public:
      TString name;
      TProxyHelper(const char *left,const char *right = 0) :
         name() {
         if (left) {
            name = left;
            if (strlen(left)&&right) name += ".";
         } 
         if (right) {
            name += right;
         }
      }
      operator const char*() { return name.Data(); };
   };
   
   
   class TProxy {
   public:
      const TString fBranchName;  // name of the branch to read
      TProxy *fParent;             // Proxy to a parent object
      
      const bool fIsMember;       // true if we proxy an unsplit data member
      const TString fDataMember;  // name of the (eventual) data member being proxied
      
      TString fClassName;         // class name of the object pointed to by the branch
      TClass *fClass;
      TStreamerElement* fElement; 
      Int_t fMemberOffset;
      bool fIsClone;
      
      TProxyDirector *fDirector; // contain pointer to TTree and entry to be read
      TBranch *fBranch;           // branch to read
      TBranch *fBranchCount;      // eventual auxiliary branch (for example holding the size)
      bool fInitialized;
      Long64_t fRead;
      
      TTree *fLastTree;
      void *fWhere;
      Int_t fOffset; // Offset inside the object
      bool fIsaPointer;
      
      virtual void Print() {
         cout << "fBranchName " << fBranchName << endl;
         //cout << "fTree " << fDirector->fTree << endl;
         cout << "fBranch " << fBranch << endl;
         if (fBranchCount) cout << "fBranchCount " << fBranchCount << endl;
      }
      
      TProxy() : fBranchName(""), fParent(0),
         fIsMember(false), fDataMember(""), fClassName(""), fClass(0), fElement(0), fMemberOffset(0),
         fIsClone(false),
         fDirector(0), fBranch(0), fBranchCount(0),
         fInitialized(false), fRead(-1), fLastTree(0), fWhere(0),
         fOffset(0), fIsaPointer(false) {
      };
      
      TProxy(TProxyDirector* boss, const char* top, const char* name = 0) : 
         fBranchName(top), 
         fParent(0),
         fIsMember(false), fDataMember(""), fClassName(""), fClass(0), fElement(0), fMemberOffset(0),
         fIsClone(false),
         fDirector(boss), fBranch(0), fBranchCount(0),
         fInitialized(false), fRead(-1), fLastTree(0), fWhere(0),
         fOffset(0), fIsaPointer(false)
         {
            if (fBranchName.Length() && fBranchName[fBranchName.Length()-1]!='.' && name) {
               ((TString&)fBranchName).Append(".");
            }
            if (name) ((TString&)fBranchName).Append(name); 
            boss->Attach(this);
         }
      
      TProxy(TProxyDirector* boss, const char *top, const char *name, const char *membername) : 
         fBranchName(top), 
         fParent(0),
         fIsMember(true), fDataMember(membername), fClassName(""), fClass(0), fElement(0), fMemberOffset(0),
         fIsClone(false),
         fDirector(boss), fBranch(0), fBranchCount(0),
         fInitialized(false), fRead(-1), fLastTree(0), fWhere(0),
         fOffset(0), fIsaPointer(false)
         {
            if (name && strlen(name)) {
               if (fBranchName.Length() && fBranchName[fBranchName.Length()-1]!='.') {
                  ((TString&)fBranchName).Append(".");
               }
               ((TString&)fBranchName).Append(name);
            }
            boss->Attach(this);
         }
      
      TProxy(TProxyDirector* boss, TProxy *parent, const char* membername) : 
         fBranchName(""), 
         fParent(parent),
         fIsMember(true), fDataMember(membername), fClassName(""), fClass(0), fElement(0), fMemberOffset(0),
         fIsClone(false),
         fDirector(boss), fBranch(0), fBranchCount(0),
         fInitialized(false), fRead(-1), fLastTree(0), fWhere(0),
         fOffset(0), fIsaPointer(false)
         {
            boss->Attach(this);
         }
      
      
      virtual ~TProxy() {};
      
      TProxy* proxy() { return this; }
      
      void reset() {
         // fprintf(stderr,"Calling reset for %s\n",fBranchName.Data());
         fWhere = 0;
         fBranch = 0;
         fBranchCount = 0;
         fRead = -1;
         fClass = 0;
         fElement = 0;
         fMemberOffset = 0;
         fIsClone = false;
         fInitialized = false;
         fLastTree = 0;
      }
      
      bool setup();

      bool IsInitialized() {
         return (fLastTree == fDirector->GetTree()) && (fLastTree);
      }
      bool IsaPointer() const {
         return fIsaPointer;
      }
      
      bool read() { 
         if (fDirector==0) return false;
         //if (fRead<2) fprintf(stderr,"read called %ld %ld\n",fRead,fDirector->fEntry);
         if (fDirector->GetReadEntry()!=fRead) {
            if (!IsInitialized()) {
               // fprintf(stderr,"%s proxy not yet initialized\n",fBranchName.Data());
            if (!setup()) {
               fprintf(stderr,"unable to initialize %s\n",fBranchName.Data());
               return false;
            }
            }
            if (fParent) fParent->read();
            else {
               fBranch->GetEntry(fDirector->GetReadEntry());
               if (fBranchCount) fBranchCount->GetEntry(fDirector->GetReadEntry());
            }
            fRead = fDirector->GetReadEntry();
         }
         //fprintf(stderr,"at the end of read where is %p\n",fWhere);
         return IsInitialized();
      }
      
      TClass *GetClass() {
         if (fDirector==0) return 0;
         if (fDirector->GetReadEntry()!=fRead) {
            if (!IsInitialized()) {
               if (!setup()) {
                  return 0;
               }
            }
         }
         return fClass;
      }
      
      // protected:
      virtual  void *GetStart(int /*i*/=0) {
         // return the address of the start of the object being proxied. Assumes
         // that setup() has been called.
         
         if (fParent) {
            fWhere = ((unsigned char*)fParent->GetStart()) + fMemberOffset;
         }
         if (IsaPointer()) {
            if (fWhere) return *(void**)fWhere;
            else return 0;
         } else {
            return fWhere;
         }
      }

      virtual void *GetClaStart(int i=0) {
         // return the address of the start of the object being proxied. Assumes
         // that setup() has been called.  Assumes the object containing this data
         // member is held in TClonesArray.

         void *tcaloc;
         char *location;

         if (fIsClone) {

            TClonesArray *tca;
            tca = (TClonesArray*)GetStart();
        
            if (tca->GetLast()<i) return 0;

            location = (char*)tca->At(i);

            return location;

         } else if (fParent) {

            tcaloc = ((unsigned char*)fParent->GetStart());
            location = (char*)fParent->GetClaStart(i);

         } else {

            tcaloc = fWhere;
            TClonesArray *tca;
            tca = (TClonesArray*)tcaloc;
        
            if (tca->GetLast()<i) return 0;

            location = (char*)tca->At(i);
         }

         if (location) location += fOffset;
         else return 0;

         if (IsaPointer()) {
            return *(void**)(location);
         } else {
            return location;
         }
     
      }
   };

   class TArrayCharProxy : public TProxy {
   public:   
      void Print() {
         TProxy::Print();
         cout << "fWhere " << fWhere << endl;
         if (fWhere) cout << "value? " << *(unsigned char*)GetStart() << endl;
      }

      TArrayCharProxy() : TProxy() {}
      TArrayCharProxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
      TArrayCharProxy(TProxyDirector *director, const char *top, const char *name) : 
         TProxy(director,top,name) {};
      TArrayCharProxy(TProxyDirector *director, const char *top, const char *name, const char *data) : 
         TProxy(director,top,name,data) {};
      TArrayCharProxy(TProxyDirector *director, TProxy *parent, const char *name) : TProxy(director,parent, name) {};
      ~TArrayCharProxy() {};

      unsigned char at(int i) {
         static unsigned char default_val;
         if (!read()) return default_val;
         // should add out-of bound test
         unsigned char* str = (unsigned char*)GetStart();
         return str[i];
      }

      unsigned char operator [](int i) {
         return at(i);
      }
   
      const char* c_str() {
         if (!read()) return "";
         return (const char*)GetStart();
      }

      operator std::string() {
         if (!read()) return "";
         return std::string((const char*)GetStart());
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
      TClaProxy(TProxyDirector *director, const char *top, const char *name, const char *data) : 
         TProxy(director,top,name,data) {};
      TClaProxy(TProxyDirector *director, TProxy *parent, const char *name) : TProxy(director,parent, name) {};
      ~TClaProxy() {};

      const TClonesArray* ptr() {
         if (!read()) return 0;
         return (TClonesArray*)GetStart();
      }   

      const TClonesArray* operator->() { return ptr(); }

   };

   template <class T> 
      class TImpProxy : public TProxy {
      public:
      void Print() {
         TProxy::Print();
         cout << "fWhere " << fWhere << endl;
         if (fWhere) cout << "value? " << *(T*)GetStart() << endl;
      }

      TImpProxy() : TProxy() {}; 
      TImpProxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
      TImpProxy(TProxyDirector *director, const char *top, const char *name) : 
         TProxy(director,top,name) {};
      TImpProxy(TProxyDirector *director, const char *top, const char *name, const char *data) : 
         TProxy(director,top,name,data) {};
      TImpProxy(TProxyDirector *director, TProxy *parent, const char *name) : TProxy(director,parent, name) {};
      ~TImpProxy() {};

      operator T() {
         if (!read()) return 0;
         return *(T*)GetStart();
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
         if (fWhere) cout << "value? " << *(T*)GetStart() << endl;
      }

      TArrayProxy() : TProxy() {}
      TArrayProxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
      TArrayProxy(TProxyDirector *director, const char *top, const char *name) : 
         TProxy(director,top,name) {};
      TArrayProxy(TProxyDirector *director, const char *top, const char *name, const char *data) : 
         TProxy(director,top,name,data) {};
      TArrayProxy(TProxyDirector *director, TProxy *parent, const char *name) : TProxy(director,parent, name) {};
      ~TArrayProxy() {};

      const T& at(int i) {
         static T default_val;
         if (!read()) return default_val;
         // should add out-of bound test
         return ((T*)GetStart())[i];
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
         if (fWhere) cout << "value? " << *(T*)GetStart() << endl;
      }

      TArray3Proxy() : TProxy() {}
      TArray3Proxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
      TArray3Proxy(TProxyDirector *director, const char *top, const char *name) : 
         TProxy(director,top,name) {};
      TArray3Proxy(TProxyDirector *director, const char *top, const char *name, const char *data) : 
         TProxy(director,top,name,data) {};
      TArray3Proxy(TProxyDirector *director, TProxy *parent, const char *name) : TProxy(director,parent, name) {};
      ~TArray3Proxy() {};

      const array_t* at(int i) {
         static array_t default_val;
         if (!read()) return &default_val;
         // should add out-of bound test
         return ((array_t**)GetStart())[i];
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
         if (fWhere) cout << "value? " << *(T*)GetStart() << endl;
      }

      TClaImpProxy() : TProxy() {}; 
      TClaImpProxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
      TClaImpProxy(TProxyDirector *director,  const char *top, const char *name) : 
         TProxy(director,top,name) {};
      TClaImpProxy(TProxyDirector *director,  const char *top, const char *name, const char *data) : 
         TProxy(director,top,name,data) {};
      TClaImpProxy(TProxyDirector *director, TProxy *parent, const char *name) : TProxy(director,parent, name) {};
      ~TClaImpProxy() {};

      const T& at(int i) {
         static T default_val;
         if (!read()) return default_val;
         if (fWhere==0) return default_val;

         T *temp = (T*)GetClaStart(i);

         if (temp) return *temp;
         else return default_val;

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
         if (fWhere) cout << "value? " << *(T*)GetStart() << endl;
      }

      TClaArrayProxy() : TProxy() {}
      TClaArrayProxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
      TClaArrayProxy(TProxyDirector *director, const char *top, const char *name) : 
         TProxy(director,top,name) {};
      TClaArrayProxy(TProxyDirector *director, const char *top, const char *name, const char *data) : 
         TProxy(director,top,name,data) {};
      TClaArrayProxy(TProxyDirector *director, TProxy *parent, const char *name) : TProxy(director,parent, name) {};
      ~TClaArrayProxy() {};

      const array_t at(int i) {
         static T default_val;
         if (!read()) return &default_val;
         if (fWhere==0) return &default_val;
     
         return (array_t)GetClaStart(i);
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
         if (fWhere) cout << "value? " << *(T*)GetStart() << endl;
      }

      TClaArray2Proxy() : TProxy() {}
      TClaArray2Proxy(TProxyDirector *director, const char *name) 
        : TProxy(director,name) {};
      TClaArray2Proxy(TProxyDirector *director, const char *top, 
                      const char *name) 
        : TProxy(director,top,name) {};
      TClaArray2Proxy(TProxyDirector *director, const char *top, 
                      const char *name, const char *data) 
        : TProxy(director,top,name,data) {};
      TClaArray2Proxy(TProxyDirector *director, TProxy *parent, 
                      const char *name) 
        : TProxy(director,parent, name) {};
      ~TClaArray2Proxy() {};

      const array_t &at(int i) {
         // might need a second param or something !?
      
         static array_t default_val;
         if (!read()) return &default_val;
         if (fWhere==0) return &default_val;
     
         T *temp = (T*)GetClaStart(i);
         if (temp) return *temp;
         else return default_val;

         // T *temp = *(T**)location;
         // return ((array_t**)temp)[i];
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
         if (fWhere) cout << "value? " << *(T*)GetStart() << endl;
      }

      TClaArray3Proxy() : TProxy() {}
      TClaArray3Proxy(TProxyDirector *director, const char *name) : TProxy(director,name) {};
      TClaArray3Proxy(TProxyDirector *director, const char *top, const char *name) : 
         TProxy(director,top,name) {};
      TClaArray3Proxy(TProxyDirector *director, const char *top, const char *name, const char *data) : 
         TProxy(director,top,name,data) {};
      TClaArray3Proxy(TProxyDirector *director, TProxy *parent, const char *name) : TProxy(director,parent, name) {};
      ~TClaArray3Proxy() {};

      const array_t* at(int i) {
         static array_t default_val;
         if (!read()) return &default_val;
         if (fWhere==0) return &default_val;
      
         T *temp = (T*)GetClaStart(i);
         if (temp) return (array_t*)temp;
         else return default_val;

         // T *temp = *(T**)location;
         // return ((array_t**)temp)[i];
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

   typedef TClaArrayProxy<double>   TClaArrayDoubleProxy;
   typedef TClaArrayProxy<float>    TClaArrayFloatProxy;
   typedef TClaArrayProxy<UInt_t>   TClaArrayUIntProxy;
   typedef TClaArrayProxy<UShort_t> TClaArrayUShortProxy;
   typedef TClaArrayProxy<UChar_t>  TClaArrayUCharProxy;
   typedef TClaArrayProxy<Int_t>    TClaArrayIntProxy;
   typedef TClaArrayProxy<UShort_t> TClaArrayShortProxy;
   //specialized ! typedef TClaArrayProxy<Char_t>  TClaArrayCharProxy;

} // namespace ROOT

#endif // TPROXY_H

/*  #ifdef __MAKECINT__ */
/*  #pragma link C++ class TClaImpProxy<unsigned int>; */



/*  #pragma link C++ class TClaArrayProxy<int>; */
/*  #pragma link C++ class TClaArrayProxy<Float_t>; */
/*  #pragma link C++ class TClaArrayProxy<double>; */

/*  #endif */

