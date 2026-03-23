// File: emulation.C

#include <iostream>
#include <cstdint> // uintptr_t
#include <cassert>
#include "TError.h"
#include "TFile.h"
#include "TBuffer.h"
#include "TObjArray.h"

// Containee type with a custom alignment (over-aligned)
struct alignas(256) Containee {
   int id;
   // some payload to make sizeof non-trivial
   double payload[7];

   Containee(int i = -1) : id(i) {}
   ~Containee() = default;

   void Streamer(TBuffer &b)
   {
      if (b.IsReading()) {
         std::cout << "Streaming Containee\n";
         // Guess the effective alignment of c's address: largest power-of-two
         // that divides the address.
         uintptr_t addr = reinterpret_cast<uintptr_t>(this);
         uintptr_t guessed = 1;
         if (addr != 0) {
            guessed = addr & (~addr + 1); // isolate lowest set bit
         }
         std::cout << "  address: " << static_cast<const void *>(this) << "  (guessed alignment: " << guessed
                   << " bytes)\n";
         if (reinterpret_cast<uintptr_t>(this) % alignof(Containee) != 0) {
            Fatal("copyContainer", "Containee object at %p does not satisfy alignment requirement of %zu\n", this,
                  alignof(Containee));
         }

         b.ReadClassBuffer(TClass::GetClass("Containee"), this);
      } else {
         b.WriteClassBuffer(TClass::GetClass("Containee"), this);
      }
   }
};

static_assert(alignof(Containee) == 256, "Containee must be 256-byte aligned");

#ifdef __ROOTCLING__
#pragma link C++ class Containee - ;
#endif

/// Check that the emulated TClass for \a className reports the expected
/// \a expectedAlignment and \a expectedSize.  Returns 0 on success or a
/// non-zero error code on failure.
int checkEmulatedClass(const char *className, std::size_t expectedAlignment, Int_t expectedSize)
{
   auto cl = TClass::GetClass(className);
   if (!cl) {
      Error("checkEmulatedClass", "Could not get TClass for %s", className);
      return 1;
   }
   if (cl->GetClassAlignment() != expectedAlignment) {
      Error("checkEmulatedClass", "TClass for %s has alignment %zu, expected %zu", className, cl->GetClassAlignment(),
            expectedAlignment);
      return 2;
   }
   if (cl->GetClassSize() != expectedSize) {
      Error("checkEmulatedClass", "TClass for %s has size %d, expected %d", className, cl->GetClassSize(),
            expectedSize);
      return 3;
   }
   return 0;
}

int readfile(const char *filename)
{
   TFile file(filename, "READ");
   if (file.IsZombie())
      return 1;
   auto c = file.Get("origContainer");
   TClass::GetClass("Container")->GetStreamerInfos()->ls();
   if (!c) {
      Error("readfile", "Could not read object from file");
      return 2;
   }

   size_t align = alignof(Containee);
   std::vector<char> vec;
   vec.resize(20);

   // Container: char(1) + padding(255) + Containee(256) + ... = 768 bytes, align=256
   if (int rc = checkEmulatedClass("Container", alignof(Containee), 768))
      return 10 + rc;

   // std::pair<int,Containee>: int(4) + padding(252) + Containee(256) = 512 bytes, align=256
   if (int rc = checkEmulatedClass("pair<int,Containee>", alignof(Containee), 512))
      return 20 + rc;

   // When emulating the Wrapper class, we start with some metadata and thus
   // the Containee members starts at offset 256, which means the total size
   // of the pair becomes 768 bytes.
   if (int rc = checkEmulatedClass("pair<int,Wrapper>", alignof(Containee), 768))
      return 30 + rc;

   return 0;
}

int emulation()
{
   return readfile("alignment_evolution.root");
}
