// vecalloc.C
// Tests round-trip TFile I/O for a Container whose data member is a
// std::vector that uses a custom over-aligned allocator.

#include <vector>
#include <cassert>
#include <iostream>
#include "TError.h"
#include "TFile.h"

template <typename T>
struct MyAlignedAllocator {
   using value_type = T;

   std::size_t alignment;

   explicit MyAlignedAllocator(std::size_t align = alignof(std::max_align_t)) : alignment(align) {}

   template <typename U>
   MyAlignedAllocator(const MyAlignedAllocator<U> &other) noexcept : alignment(other.alignment)
   {
   }

   T *allocate(std::size_t n)
   {
      std::size_t bytes = n * sizeof(T);
      std::size_t effectiveAlign = alignment < alignof(T) ? alignof(T) : alignment;
      // Round up to a multiple of effectiveAlign (required by aligned operator new)
      bytes = (bytes + effectiveAlign - 1) & ~(effectiveAlign - 1);
      return static_cast<T *>(::operator new(bytes, std::align_val_t{effectiveAlign}));
   }

   void deallocate(T *p, std::size_t n) noexcept
   {
      std::size_t bytes = n * sizeof(T);
      std::size_t effectiveAlign = alignment < alignof(T) ? alignof(T) : alignment;
      bytes = (bytes + effectiveAlign - 1) & ~(effectiveAlign - 1);
      ::operator delete(p, bytes, std::align_val_t{effectiveAlign});
   }

   template <typename U>
   bool operator==(const MyAlignedAllocator<U> &other) const noexcept
   {
      return alignment == other.alignment;
   }
   template <typename U>
   bool operator!=(const MyAlignedAllocator<U> &other) const noexcept
   {
      return !(*this == other);
   }
};

struct Container {
   std::vector<int, MyAlignedAllocator<int>> data;

   Container() = default;
   Container(std::initializer_list<int> vals) : data(vals) {}
   template <typename Iter>
   Container(Iter first, Iter last) : data(first, last)
   {
   }

   ClassDefNV(Container, 1)
};

#ifdef __ROOTCLING__
#pragma link C++ class Container + ;
#endif

static const char *kFileName = "vecalloc_test.root";
static const char *kObjName = "cont";

// Known values written into the Container
static const std::vector<int> kExpected = {10, 20, 30, 42, 99};

int writefile()
{
   Container c(kExpected.begin(), kExpected.end());
   std::cout << "Writing Container with data:";
   for (int v : c.data)
      std::cout << ' ' << v;
   std::cout << '\n';

   TFile f(kFileName, "RECREATE");
   if (f.IsZombie()) {
      Error("writefile", "Cannot open %s for writing", kFileName);
      return 1;
   }
   f.WriteObject(&c, kObjName);
   f.Write();
   return 0;
}

int readfile()
{
   TFile f(kFileName, "READ");
   if (f.IsZombie()) {
      Error("readfile", "Cannot open %s for reading", kFileName);
      return 1;
   }

   Container *c = f.Get<Container>(kObjName);
   if (!c) {
      Error("readfile", "Failed to read Container '%s' from file", kObjName);
      return 1;
   }

   std::cout << "Read back Container with data:";
   for (int v : c->data)
      std::cout << ' ' << v;
   std::cout << '\n';

   // Verify size
   if (c->data.size() != kExpected.size()) {
      Error("readfile", "Size mismatch: expected %zu, got %zu", kExpected.size(), c->data.size());
      return 1;
   }

   // Verify values
   for (std::size_t i = 0; i < kExpected.size(); ++i) {
      if (c->data[i] != kExpected[i]) {
         Error("readfile", "Value mismatch at index %zu: expected %d, got %d", i, kExpected[i], c->data[i]);
         return 1;
      }
   }

   std::cout << "All values verified successfully.\n";
   return 0;
}

int vecalloc()
{
   if (int ret = writefile())
      return ret;
   return readfile();
}

int main()
{
   return vecalloc();
}
