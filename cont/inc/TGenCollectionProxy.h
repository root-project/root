// @(#)root/cont:$Name:  $:$Id: TGenCollectionProxy.h,v 1.11 2006/05/19 07:30:04 brun Exp $
// Author: Markus Frank  28/10/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TGenCollectionProxy
#define ROOT_TGenCollectionProxy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGenCollectionProxy
//
// Proxy around an arbitrary container, which implements basic
// functionality and iteration.
//
// In particular this is used to implement splitting and abstract
// element access of any container. Access to compiled code is necessary
// to implement the abstract iteration sequence and functionality like
// size(), clear(), resize(). resize() may be a void operation.
//
//////////////////////////////////////////////////////////////////////////

#include "TVirtualCollectionProxy.h"
#include "TCollectionProxy.h"
#include <typeinfo>
#include <string>

class TGenCollectionProxy
   : public TVirtualCollectionProxy
{

   // Friend declaration
   friend class TCollectionProxy;

public:

#ifdef R__HPUX
   typedef const type_info&      Info_t;
#else
   typedef const std::type_info& Info_t;
#endif

   enum {
      // Those 'bits' are used in conjunction with CINT's bit to store the 'type'
      // info into one int
      kBIT_ISSTRING   = 0x20000000,  // We can optimized a value operation when the content are strings
      kBIT_ISTSTRING  = 0x40000000,
      kBOOL_t = 21
   };


   /** @class TGenCollectionProxy::Value TGenCollectionProxy.h TGenCollectionProxy.h
    *
    * Small helper to describe the Value_type or the key_type
    * of an STL container.
    *
    * @author  M.Frank
    * @version 1.0
    * @date    10/10/2004
    */
   struct Value  {
      ROOT::NewFunc_t fCtor;      // Method cache for containee constructor
      ROOT::DesFunc_t fDtor;      // Method cache for containee destructor
      ROOT::DelFunc_t fDelete;    // Method cache for containee delete
      unsigned int    fCase;      // type of data of Value_type
      TClassRef       fType;      // TClass reference of Value_type in collection
      EDataType       fKind;      // kind of ROOT-fundamental type
      size_t          fSize;      // fSize of the contained object

      // Copy constructor
      Value(const Value& inside);
      // Initializing constructor
      Value(const std::string& info);
      // Delete individual item from STL container
      void DeleteItem(void* ptr);
   };

   /**@class StreamHelper
    *
    * Helper class to facilitate I/O
    *
    * @author  M.Frank
    * @version 1.0
    * @date    10/10/2004
    */
   union StreamHelper  {
      Bool_t       boolean;
      Char_t       s_char;
      Short_t      s_short;
      Int_t        s_int;
      Long_t       s_long;
      Long64_t     s_longlong;
      Float_t      flt;
      Double_t     dbl;
      UChar_t      u_char;
      UShort_t     u_short;
      UInt_t       u_int;
      ULong_t      u_long;
      ULong64_t    u_longlong;
      void*        p_void;
      void**       pp_void;
      char*        kchar;
      TString*     tstr;
      void* ptr()  {
         return *(&this->p_void);
      }
      std::string* str()  {
         return (std::string*)this;
      }
      const char* c_str()  {
         return ((std::string*)this)->c_str();
      }
      const char* c_pstr()  {
         return (*(std::string**)this)->c_str();
      }
      void set(void* p)  {
         *(&this->p_void) = p;
      }
      void read_std_string(TBuffer& b) {
         TString s;
         s.Streamer(b);
         ((std::string*)this)->assign(s.Data());
      }
      void* read_tstring(TBuffer& b)  {
         *((TString*)this) = "";
         ((TString*)this)->Streamer(b);
         return this;
      }
      void read_std_string_pointer(TBuffer& b) {
         TString s;
         std::string* str = (std::string*)ptr();
         if (!str) str = new std::string();
         s.Streamer(b);
         *str = s;
         set(str);
      }
      void write_std_string_pointer(TBuffer& b)  {
         const char* c;
         if (ptr()) {
            std::string* strptr = (*(std::string**)this);
            c = (const char*)(strptr->c_str());
         } else c = "";
         TString(c).Streamer(b);
      }
      void read_any_object(Value* v, TBuffer& b)  {
         void* p = ptr();
         if ( p )  {
            if ( v->fDelete )  {    // Compiled content: call Destructor
               (*v->fDelete)(p);
            }
            else if ( v->fType )  { // Emulated content: call TClass::Delete
               v->fType->Destructor(p);
            }
            else if ( v->fDtor )  {
               (*v->fDtor)(p);
               ::operator delete(p);
            }
            else  {
               ::operator delete(p);
            }
         }
         set( b.ReadObjectAny(v->fType) );
      }

      void read_tstring_pointer(Bool_t vsn3, TBuffer& b)  {
         TString* s = (TString*)ptr();
         if ( vsn3 )  {
            if ( !s ) s = new TString();
            s->Replace(0, s->Length(), 0, 0);
            s->Streamer(b);
            set(s);
            return;
         }
         if ( s ) delete s;
         set( b.ReadObjectAny(TString::Class()) );
      }
      void write_tstring_pointer(TBuffer& b)  {
         b.WriteObjectAny(ptr(), TString::Class());
      }
   };

   /** @class TGenCollectionProxy::Method TGenCollectionProxy.h TGenCollectionProxy.h
    *
    * Small helper to execute (compiler) generated function for the
    * access to STL or other containers.
    *
    * @author  M.Frank
    * @version 1.0
    * @date    10/10/2004
    */
   struct Method  {
      typedef void* (*Call_t)(void*);
      Call_t call;
      Method() : call(0)                       {      }
      Method(Call_t c) : call(c)               {      }
      Method(const Method& m) : call(m.call)   {      }
      void* invoke(void* obj) const { return (*call)(obj); }
   };

protected:
   typedef ROOT::TCollectionProxyInfo::Environ<char[64]> Env_t;
   typedef std::vector<Env_t*>     Proxies_t;

   std::string   fName;      // Name of the class being proxied.
   Bool_t        fPointers;  // Flag to indicate if containee has pointers (key or value)
   Method        fClear;     // Method cache for container accessors: clear container
   Method        fSize;      // Container accessors: size of container
   Method        fResize;    // Container accessors: resize container
   Method        fFirst;     // Container accessors: generic iteration: first
   Method        fNext;      // Container accessors: generic iteration: next
   Method        fConstruct; // Container accessors: block construct
   Method        fDestruct;  // Container accessors: block destruct
   Method        fFeed;      // Container accessors: block feed
   Method        fCollect;   // Method to collect objects from container
   Value*        fValue;     // Descriptor of the container value type
   Value*        fVal;       // Descriptor of the Value_type
   Value*        fKey;       // Descriptor of the key_type
   Env_t*        fEnv;       // Address of the currently proxied object
   int           fValOffset; // Offset from key to value (in maps)
   int           fValDiff;   // Offset between two consecutive value_types (memory layout).
   Proxies_t     fProxyList; // Stack of recursive proxies
   Proxies_t     fProxyKept; // Optimization: Keep proxies once they were created
   int           fSTL_type;  // STL container type
   Info_t        fTypeinfo;  // Type information

   // Late initialization of collection proxy
   TGenCollectionProxy* Initialize() const;
   // Some hack to avoid const-ness.
   virtual TGenCollectionProxy* InitializeEx();
   // Call to delete/destruct individual contained item.
   virtual void DeleteItem(Bool_t force, void* ptr) const;
   // Allow to check function pointers.
   void CheckFunctions()  const;

public:

   // Virtual copy constructor.
   virtual TVirtualCollectionProxy* Generate() const;

   // Copy constructor.
   TGenCollectionProxy(const TGenCollectionProxy& copy);

   // Initializing constructor
   TGenCollectionProxy(Info_t typ, size_t iter_size);
   TGenCollectionProxy(const ROOT::TCollectionProxyInfo &info);

   // Standard destructor.
   virtual ~TGenCollectionProxy();

   // Return a pointer to the TClass representing the container.
   virtual TClass *GetCollectionClass();

   // Return the sizeof the collection object.
   virtual UInt_t Sizeof() const;

   // Push new proxy environment.
   virtual void PushProxy(void *objstart);

   // Pop old proxy environment.
   virtual void PopProxy();

   // Return true if the content is of type 'pointer to'.
   virtual Bool_t HasPointers() const;

   // Return a pointer to the TClass representing the content.
   virtual TClass *GetValueClass();

   // Set pointer to the TClass representing the content.
   virtual void SetValueClass(TClass *newcl);

   // If the content is a simple numerical value, return its type (see TDataType).
   virtual EDataType GetType();

   // Return the address of the value at index 'idx'.
   virtual void *At(UInt_t idx);

   // Clear the container.
   virtual void Clear(const char *opt = "");

   // Resize the container.
   virtual void Resize(UInt_t n, Bool_t force_delete);

   // Return the current size of the container.
   virtual UInt_t Size() const;

   // Block allocation of containees.
   virtual void* Allocate(UInt_t n, Bool_t forceDelete);

   // Block commit of containees.
   virtual void Commit(void* env);

   // Streamer function.
   virtual void Streamer(TBuffer &refBuffer);

   // Streamer I/O overload.
   virtual void Streamer(TBuffer &refBuffer, void *pObject, int siz);

   // TClassStreamer I/O overload.
   virtual void operator()(TBuffer &refBuffer, void *pObject);
};

template <typename T>
struct AnyCollectionProxy : public TGenCollectionProxy  {
   AnyCollectionProxy()
      : TGenCollectionProxy(typeid(T::Cont_t),sizeof(T::Iter_t))
   {
      // Constructor.
      fValDiff        = sizeof(T::Value_t);
      fValOffset      = T::value_offset();
      fSize.call      = T::size;
      fResize.call    = T::resize;
      fNext.call      = T::next;
      fFirst.call     = T::first;
      fClear.call     = T::clear;
      fConstruct.call = T::construct;
      fDestruct.call  = T::destruct;
      fFeed.call      = T::feed;
      CheckFunctions();
   }
   virtual ~AnyCollectionProxy() {  }
};

#endif

