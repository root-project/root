// @(#)root/io:$Id$
// Author: Markus Frank  28/10/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TEmulatedCollectionProxy
#define ROOT_TEmulatedCollectionProxy

#include "TGenCollectionProxy.h"
#include "ROOT/BitUtils.hxx"

#include <type_traits>
#include <vector>

class TEmulatedCollectionProxy : public TGenCollectionProxy  {

   // Friend declaration
   friend class TCollectionProxy;

public:
   /// Storage type whose alignment matches \a Align bytes.
   /// Used to instantiate std::vector specializations with guaranteed buffer alignment.
   template <std::size_t Align>
   struct alignas(Align) AlignedStorage {
      char data[Align] = {};
   };

   // Convenience vector aliases for each supported alignment.
   using Cont1_t    = std::vector<AlignedStorage<   1>>;
   using Cont2_t    = std::vector<AlignedStorage<   2>>;
   using Cont4_t    = std::vector<AlignedStorage<   4>>;
   using Cont8_t    = std::vector<AlignedStorage<   8>>;
   using Cont16_t   = std::vector<AlignedStorage<  16>>;
   using Cont32_t   = std::vector<AlignedStorage<  32>>;
   using Cont64_t   = std::vector<AlignedStorage<  64>>;
   using Cont128_t  = std::vector<AlignedStorage< 128>>;
   using Cont256_t  = std::vector<AlignedStorage< 256>>;
   using Cont512_t  = std::vector<AlignedStorage< 512>>;
   using Cont1024_t = std::vector<AlignedStorage<1024>>;
   using Cont2048_t = std::vector<AlignedStorage<2048>>;
   using Cont4096_t = std::vector<AlignedStorage<4096>>;

   // Canonical container type (used for sizeof/typeid; actual alignment is
   // selected at runtime via the alignment switch in each method).
   using Cont_t  = std::vector<char>;
   using PCont_t = Cont_t *;

   /// Invoke \a fn(typed_ptr, elemSize) where typed_ptr is the container
   /// pointer cast to the correct AlignedStorage<N>* for the value class
   /// alignment.  \a fn receives the element size (N) as a second argument
   /// so it can convert byte counts to element counts.
   template <typename F>
   void WithCont(void *obj, F &&fn) const
   {
      auto *vcl = GetValueClass();
      std::size_t align = alignof(std::max_align_t);
      if (!fKey && (fVal->fCase & kIsPointer)) {
         // If the collection contains pointers, we need to use the alignment of a pointer, not of the value class.
         align = alignof(void*);
      } else if (vcl) {
         assert(ROOT::Internal::IsValidAlignment(vcl->GetClassAlignment()));
         align = vcl->GetClassAlignment();
      } else {
         switch( int(fVal->fKind) ) {
            case kChar_t:
            case kUChar_t:  align = alignof(char); break;
            case kShort_t:
            case kUShort_t: align = alignof(short); break;
            case kInt_t:
            case kUInt_t:   align = alignof(int); break;
            case kLong_t:
            case kULong_t:  align = alignof(long); break;
            case kLong64_t:
            case kULong64_t:align = alignof(long long); break;
            case kFloat16_t:
            case kFloat_t:  align = alignof(float); break;
            case kDouble32_t:
            case kDouble_t: align = alignof(double); break;
            default:
               Fatal("TEmulatedCollectionProxy::WithCont", "Unsupported value type %d for value class %s", fVal->fKind,
                     vcl ? vcl->GetName() : "<unknown>");
         }
      }
      switch (align) {
         // When adding new cases here, also update the static_assert in TClingUtils.cxx
         // to explicitly allow the new alignment and to update the error message accordingly.
         case 4096: fn(reinterpret_cast<Cont4096_t*>(obj), std::size_t(4096)); break;
         case 2048: fn(reinterpret_cast<Cont2048_t*>(obj), std::size_t(2048)); break;
         case 1024: fn(reinterpret_cast<Cont1024_t*>(obj), std::size_t(1024)); break;
         case  512: fn(reinterpret_cast<Cont512_t *>(obj), std::size_t( 512)); break;
         case  256: fn(reinterpret_cast<Cont256_t *>(obj), std::size_t( 256)); break;
         case  128: fn(reinterpret_cast<Cont128_t *>(obj), std::size_t( 128)); break;
         case   64: fn(reinterpret_cast<Cont64_t  *>(obj), std::size_t(  64)); break;
         case   32: fn(reinterpret_cast<Cont32_t  *>(obj), std::size_t(  32)); break;
         case   16: fn(reinterpret_cast<Cont16_t  *>(obj), std::size_t(  16)); break;
         case    8: fn(reinterpret_cast<Cont8_t   *>(obj), std::size_t(   8)); break;
         case    4: fn(reinterpret_cast<Cont4_t   *>(obj), std::size_t(   4)); break;
         case    2: fn(reinterpret_cast<Cont2_t   *>(obj), std::size_t(   2)); break;
         case    1: fn(reinterpret_cast<Cont1_t   *>(obj), std::size_t(   1)); break;
         default:
            Fatal("TEmulatedCollectionProxy::WithCont", "Unsupported alignment %zu for value class %s",
                  align, vcl ? vcl->GetName() : "<unknown>");
      }
   }
protected:

   // Some hack to avoid const-ness
   TGenCollectionProxy* InitializeEx(Bool_t silent) override;

   // Object input streamer
   void ReadItems(int nElements, TBuffer &b);

   // Object output streamer
   void WriteItems(int nElements, TBuffer &b);

   // Shrink the container
   void Shrink(UInt_t nCurr, UInt_t left, Bool_t force);

   // Expand the container
   void Expand(UInt_t nCurr, UInt_t left);

private:
   TEmulatedCollectionProxy &operator=(const TEmulatedCollectionProxy &); // Not implemented.

public:
   // Virtual copy constructor
   TVirtualCollectionProxy* Generate() const override;

   // Copy constructor
   TEmulatedCollectionProxy(const TEmulatedCollectionProxy& copy);

   // Initializing constructor
   TEmulatedCollectionProxy(const char* cl_name, Bool_t silent);

   // Standard destructor
   ~TEmulatedCollectionProxy() override;

   // Virtual constructor
   void *New() const override
   {
      void *mem = ::operator new(sizeof(Cont_t));
      WithCont(mem, [](auto *c, std::size_t) { new (c) std::decay_t<decltype(*c)>(); });
      return mem;
   }

   // Virtual in-place constructor
   void *New(void *memory) const override
   {
      WithCont(memory, [](auto *c, std::size_t) { new (c) std::decay_t<decltype(*c)>(); });
      return memory;
   }

   // Virtual constructor
   TClass::ObjectPtr NewObject() const override { return {New(), nullptr}; }

   // Virtual in-place constructor
   TClass::ObjectPtr NewObject(void *memory) const override { return {New(memory), nullptr}; }

   // Virtual array constructor
   void *NewArray(Int_t nElements) const override
   {
      void *arr = ::operator new(nElements * sizeof(Cont_t));
      for (Int_t i = 0; i < nElements; ++i)
         WithCont(static_cast<char *>(arr) + i * sizeof(Cont_t),
                  [](auto *c, std::size_t) { new (c) std::decay_t<decltype(*c)>(); });
      return arr;
   }

   // Virtual in-place array constructor
   void *NewArray(Int_t nElements, void *memory) const override
   {
      for (Int_t i = 0; i < nElements; ++i)
         WithCont(static_cast<char *>(memory) + i * sizeof(Cont_t),
                  [](auto *c, std::size_t) { new (c) std::decay_t<decltype(*c)>(); });
      return memory;
   }

   // Virtual array constructor
   TClass::ObjectPtr NewObjectArray(Int_t nElements) const override { return {NewArray(nElements), nullptr}; }

   // Virtual in-place array constructor
   TClass::ObjectPtr NewObjectArray(Int_t nElements, void *memory) const override
   {
      return {NewArray(nElements, memory), nullptr};
   }

   // Virtual destructor
   void  Destructor(void* p, Bool_t dtorOnly = kFALSE) const override;

   // Virtual array destructor
   void  DeleteArray(void* p, Bool_t dtorOnly = kFALSE) const override;

   // TVirtualCollectionProxy overload: Return the sizeof the collection object.
   UInt_t Sizeof() const override { return sizeof(Cont_t); }

   // Return the address of the value at index 'idx'
   void *At(UInt_t idx) override;

   // Clear the container
   void Clear(const char *opt = "") override;

   // Resize the container
   void Resize(UInt_t n, Bool_t force_delete) override;

   // Return the current size of the container
   UInt_t Size() const override;

   // Block allocation of containees
   void* Allocate(UInt_t n, Bool_t forceDelete) override;

   // Block commit of containees
   void Commit(void* env) override;

   // Insert data into the container where data is a C-style array of the actual type contained in the collection
   // of the given size.   For associative container (map, etc.), the data type is the pair<key,value>.
   void  Insert(const void *data, void *container, size_t size) override;

   // Read portion of the streamer
   void ReadBuffer(TBuffer &buff, void *pObj) override;
   void ReadBuffer(TBuffer &buff, void *pObj, const TClass *onfile) override;

   // Streamer for I/O handling
   void Streamer(TBuffer &refBuffer) override;

   // Streamer I/O overload
   void Streamer(TBuffer &buff, void *pObj, int siz) override
   {
      TGenCollectionProxy::Streamer(buff,pObj,siz);
   }

   // Check validity of the proxy itself
   Bool_t IsValid() const;
};


namespace ROOT::Internal::TEmulatedProxyHelpers {
inline void PrintWriteStlWithoutProxyMsg(const char *where, const char *clName, const char *BranchName)
{
   const char *writeStlWithoutProxyMsg =
      "The class requested (%s) for the branch \"%s\" "
      "is an instance of an stl collection and does not have a compiled CollectionProxy. "
      "Please generate the dictionary for this collection (%s) to avoid writing corrupted data.";
   // This error message is repeated several times in the code. We write it once.
   Error(where, writeStlWithoutProxyMsg, clName, BranchName, clName);
}

inline bool HasEmulatedProxy(TClass *cl){
   return cl && cl->GetCollectionProxy() && dynamic_cast<TEmulatedCollectionProxy *>(cl->GetCollectionProxy());
}
} // namespace ROOT::Internal::TEmulatedProxyHelpers


#endif
