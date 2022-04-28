// @(#)root/io:$Id$
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

#include "TBuffer.h"

#include "TVirtualCollectionProxy.h"

#include "TCollectionProxyInfo.h"

#include <atomic>
#include <string>
#include <map>
#include <cstdlib>
#include <vector>

class TObjArray;
class TCollectionProxyFactory;

class TGenCollectionProxy
   : public TVirtualCollectionProxy
{

   // Friend declaration
   friend class TCollectionProxyFactory;

public:

#ifdef R__HPUX
   typedef const std::type_info&      Info_t;
#else
   typedef const std::type_info& Info_t;
#endif

   enum {
      // Those 'bits' are used in conjunction with CINT's bit to store the 'type'
      // info into one int
      kBIT_ISSTRING   = 0x20000000,  // We can optimized a value operation when the content are strings
      kBIT_ISTSTRING  = 0x40000000
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
      ROOT::NewFunc_t fCtor;       ///< Method cache for containee constructor
      ROOT::DesFunc_t fDtor;       ///< Method cache for containee destructor
      ROOT::DelFunc_t fDelete;     ///< Method cache for containee delete
      UInt_t          fCase;       ///< type of data of Value_type
      UInt_t          fProperties; ///< Additional properties of the value type (kNeedDelete)
      TClassRef       fType;       ///< TClass reference of Value_type in collection
      EDataType       fKind;       ///< kind of ROOT-fundamental type
      size_t          fSize;       ///< fSize of the contained object

      // Default copy constructor has the correct implementation.

      // Initializing constructor
      Value(const std::string& info, Bool_t silent, size_t hint_pair_offset = 0, size_t hint_pair_size = 0);
      // Delete individual item from STL container
      void DeleteItem(void* ptr);

      Bool_t IsValid();
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
         std::string* str2 = (std::string*)ptr();
         if (!str2) str2 = new std::string();
         s.Streamer(b);
         *str2 = s;
         set(str2);
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
            else s->Clear();
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
   class Method  {
   public:
      typedef void* (*Call_t)(void*);
      Call_t call;
      Method() : call(0)                       {      }
      Method(Call_t c) : call(c)               {      }
      Method(const Method& m) : call(m.call)   {      }
      Method &operator=(const Method& m) { call = m.call; return *this; }
      void* invoke(void* obj) const { return (*call)(obj); }
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
   class Method0  {
   public:
      typedef void* (*Call_t)();
      Call_t call;
      Method0() : call(0)                       {      }
      Method0(Call_t c) : call(c)               {      }
      Method0(const Method0& m) : call(m.call)   {      }
      Method0 &operator=(const Method0& m) { call = m.call; return *this; }
      void* invoke() const { return (*call)(); }
   };

   /** @class TGenCollectionProxy::TStaging
    *
    * Small helper to stage the content of an associative
    * container when reading and before inserting it in the
    * actual collection.
    *
    * @author  Ph.Canal
    * @version 1.0
    * @date    20/08/2010
    */
   class TStaging  {
      void   *fTarget;   ///< Pointer to the collection we are staging for.
      void   *fContent;  ///< Pointer to the content
      size_t  fReserved; ///< Amount of space already reserved.
      size_t  fSize;     ///< Number of elements
      size_t  fSizeOf;   ///< size of each elements

      TStaging(const TStaging&);            ///< Not implemented.
      TStaging &operator=(const TStaging&); ///< Not implemented.

   public:
      TStaging(size_t size, size_t size_of) : fTarget(0), fContent(0), fReserved(0), fSize(size), fSizeOf(size_of)
      {
         // Usual constructor.  Reserves the required number of elements.
         fReserved = fSize;
         fContent = ::malloc(fReserved * fSizeOf);
      }

      ~TStaging() {
         // Usual destructor
         ::free(fContent);
      }

      void   *GetContent() {
         // Return the location of the array of content.
         return fContent;
      }
      void   *GetEnd() {
         // Return the 'end' of the array of content.
         return ((char*)fContent) + fSize*fSizeOf;
      }
      size_t  GetSize() {
         // Return the number of elements.
         return fSize;
      }
      void   *GetTarget() {
         // Get the address of the collection we are staging for.
         return fTarget;
      }
      void    Resize(size_t nelement) {
         if (fReserved < nelement) {
            fReserved = nelement;
            fContent = ::realloc(fContent,fReserved * fSizeOf);
         }
         fSize = nelement;
      }
      void SetTarget(void *target) {
         // Set the collection we are staging for.
         fTarget = target;
      }
   };

protected:
   typedef ROOT::Detail::TCollectionProxyInfo::Environ<char[64]> Env_t;
   typedef ROOT::Detail::TCollectionProxyInfo::EnvironBase EnvironBase_t;
   typedef std::vector<TStaging*>          Staged_t;  ///< Collection of pre-allocated staged array for associative containers.
   typedef std::vector<EnvironBase_t*>     Proxies_t;
   mutable TObjArray *fReadMemberWise;                                   ///< Array of bundle of TStreamerInfoActions to stream out (read)
   mutable std::map<std::string, TObjArray*> *fConversionReadMemberWise; ///< Array of bundle of TStreamerInfoActions to stream out (read) derived from another class.
   mutable TStreamerInfoActions::TActionSequence *fWriteMemberWise;
   typedef void (*Sizing_t)(void *obj, size_t size);
   typedef void* (*Feedfunc_t)(void *from, void *to, size_t size);
   typedef void* (*Collectfunc_t)(void *from, void *to);
   typedef void* (*ArrIterfunc_t)(void *from, size_t size);

   std::string   fName;      ///< Name of the class being proxied.
   Bool_t        fPointers;  ///< Flag to indicate if containee has pointers (key or value)
   Method        fClear;     ///< Method cache for container accessors: clear container
   Method        fSize;      ///< Container accessors: size of container
   Sizing_t      fResize;    ///< Container accessors: resize container
   Method        fFirst;     ///< Container accessors: generic iteration: first
   Method        fNext;      ///< Container accessors: generic iteration: next
   ArrIterfunc_t fConstruct; ///< Container accessors: block construct
   Sizing_t      fDestruct;  ///< Container accessors: block destruct
   Feedfunc_t    fFeed;      ///< Container accessors: block feed
   Collectfunc_t fCollect;   ///< Method to collect objects from container
   Method0       fCreateEnv; ///< Method to allocate an Environment holder.
   std::atomic<Value*> fValue;     ///< Descriptor of the container value type
   Value*        fVal;       ///< Descriptor of the Value_type
   Value*        fKey;       ///< Descriptor of the key_type
   EnvironBase_t*fEnv;       ///< Address of the currently proxied object
   int           fValOffset; ///< Offset from key to value (in maps)
   int           fValDiff;   ///< Offset between two consecutive value_types (memory layout).
   Proxies_t     fProxyList; ///< Stack of recursive proxies
   Proxies_t     fProxyKept; ///< Optimization: Keep proxies once they were created
   Staged_t      fStaged;    ///< Optimization: Keep staged array once they were created
   int           fSTL_type;  ///< STL container type
   Info_t        fTypeinfo;  ///< Type information
   TClass*       fOnFileClass; ///< On file class

   CreateIterators_t    fFunctionCreateIterators;
   CopyIterator_t       fFunctionCopyIterator;
   Next_t               fFunctionNextIterator;
   DeleteIterator_t     fFunctionDeleteIterator;
   DeleteTwoIterators_t fFunctionDeleteTwoIterators;

   // Late initialization of collection proxy
   TGenCollectionProxy* Initialize(Bool_t silent) const;
   // Some hack to avoid const-ness.
   virtual TGenCollectionProxy* InitializeEx(Bool_t silent);
   // Call to delete/destruct individual contained item.
   virtual void DeleteItem(Bool_t force, void* ptr) const;
   // Allow to check function pointers.
   void CheckFunctions()  const;

private:
   TGenCollectionProxy(); // not implemented on purpose.

public:

   // Virtual copy constructor.
   virtual TVirtualCollectionProxy* Generate() const override;

   // Copy constructor.
   TGenCollectionProxy(const TGenCollectionProxy& copy);

private:
   // Assignment operator
   TGenCollectionProxy &operator=(const TGenCollectionProxy&); // Not Implemented

public:
   // Initializing constructor
   TGenCollectionProxy(Info_t typ, size_t iter_size);
   TGenCollectionProxy(const ROOT::Detail::TCollectionProxyInfo &info, TClass *cl);

   // Standard destructor.
   virtual ~TGenCollectionProxy();

   // Reset the info gathered from StreamerInfos and value's TClass.
   Bool_t Reset() override;

   // Return a pointer to the TClass representing the container.
   TClass *GetCollectionClass() const override;

   // Return the type of collection see TClassEdit::ESTLType
   Int_t   GetCollectionType() const override;

   // Return the offset between two consecutive value_types (memory layout).
   ULong_t   GetIncrement() const override;

   // Return the sizeof the collection object.
   UInt_t Sizeof() const override;

   // Push new proxy environment.
   void PushProxy(void *objstart) override;

   // Pop old proxy environment.
   void PopProxy() override;

   // Return true if the content is of type 'pointer to'.
   Bool_t HasPointers() const override;

   // Return a pointer to the TClass representing the content.
   TClass *GetValueClass() const override;

   // If the content is a simple numerical value, return its type (see TDataType).
   EDataType GetType() const override;

   // Return the address of the value at index 'idx'.
   void *At(UInt_t idx) override;

   // Clear the container.
   void Clear(const char *opt = "") override;

   // Resize the container.
   virtual void Resize(UInt_t n, Bool_t force_delete);

   // Return the current size of the container.
   UInt_t Size() const override;

   // Block allocation of containees.
   void* Allocate(UInt_t n, Bool_t forceDelete) override;

   // Insert data into the container where data is a C-style array of the actual type contained in the collection
   // of the given size.   For associative container (map, etc.), the data type is the pair<key,value>.
   void  Insert(const void *data, void *container, size_t size) override;

   // Block commit of containees.
   void Commit(void* env) override;

   // Streamer function.
   virtual void Streamer(TBuffer &refBuffer);

   // Streamer I/O overload.
   virtual void Streamer(TBuffer &refBuffer, void *pObject, int siz);

   // TClassStreamer I/O overload.
   virtual void operator()(TBuffer &refBuffer, void *pObject);

   // Routine to read the content of the buffer into 'obj'.
   virtual void ReadBuffer(TBuffer &b, void *obj);
   virtual void ReadBuffer(TBuffer &b, void *obj, const TClass *onfileClass);

   virtual void SetOnFileClass( TClass* cl ) { fOnFileClass = cl; }
   virtual TClass* GetOnFileClass() const { return fOnFileClass; }

   // MemberWise actions
   TStreamerInfoActions::TActionSequence *GetConversionReadMemberWiseActions(TClass *oldClass, Int_t version) override;
   TStreamerInfoActions::TActionSequence *GetReadMemberWiseActions(Int_t version) override;
   TStreamerInfoActions::TActionSequence *GetWriteMemberWiseActions() override;

   // Set of functions to iterate easily through the collection

   CreateIterators_t GetFunctionCreateIterators(Bool_t read = kTRUE) override;
   // typedef void (*CreateIterators_t)(void *collection, void **begin_arena, void **end_arena);
   // begin_arena and end_arena should contain the location of a memory arena of size fgIteratorSize.
   // If the collection iterator are of that size or less, the iterators will be constructed in place in those location (new with placement)
   // Otherwise the iterators will be allocated via a regular new and their address returned by modifying the value of begin_arena and end_arena.

   CopyIterator_t GetFunctionCopyIterator(Bool_t read = kTRUE) override;
   // typedef void* (*CopyIterator_t)(void **dest, const void *source);
   // Copy the iterator source, into dest.   dest should contain the location of a memory arena of size fgIteratorSize.
   // If the collection iterator is of that size or less, the iterator will be constructed in place in this location (new with placement)
   // Otherwise the iterator will be allocated via a regular new.
   // The actual address of the iterator is returned in both case.

   Next_t GetFunctionNext(Bool_t read = kTRUE) override;
   // typedef void* (*Next_t)(void *iter, const void *end);
   // iter and end should be pointers to respectively an iterator to be incremented and the result of collection.end()
   // If the iterator has not reached the end of the collection, 'Next' increment the iterator 'iter' and return 0 if
   // the iterator reached the end.
   // If the end was not reached, 'Next' returns the address of the content pointed to by the iterator before the
   // incrementation ; if the collection contains pointers, 'Next' will return the value of the pointer.

   DeleteIterator_t GetFunctionDeleteIterator(Bool_t read = kTRUE) override;
   DeleteTwoIterators_t GetFunctionDeleteTwoIterators(Bool_t read = kTRUE) override;
   // typedef void (*DeleteIterator_t)(void *iter);
   // typedef void (*DeleteTwoIterators_t)(void *begin, void *end);
   // If the size of the iterator is greater than fgIteratorArenaSize, call delete on the addresses,
   // Otherwise just call the iterator's destructor.

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
      fResize         = T::resize;
      fNext.call      = T::next;
      fFirst.call     = T::first;
      fClear.call     = T::clear;
      fConstruct      = T::construct;
      fDestruct       = T::destruct;
      fFeed           = T::feed;
      CheckFunctions();
   }
   virtual ~AnyCollectionProxy() {  }
};

#endif

