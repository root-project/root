// File: evolution.C
// Demonstrates a containee with custom alignment embedded directly inside a
// container (no heap allocation).  The compiler is responsible for satisfying
// the alignment requirement of the embedded object, inserting padding before it
// if necessary.

#include <iostream>
#include <cstdint> // uintptr_t
#include <cassert>
#include "TError.h"
#include "TFile.h"

// Containee type with a custom alignment (over-aligned)
struct alignas(256) Containee {
   int id;
   // some payload to make sizeof non-trivial
   double payload[7];

   Containee(int i = -1) : id(i) {}
   ~Containee() = default;
};

struct Wrapper {
   Containee c;
   Wrapper(int i = -1) : c(i) {}
   ~Wrapper() = default;
};

static_assert(alignof(Containee) == 256, "Containee must be 256-byte aligned");

// Container that embeds a single Containee object directly as a data member.
// The compiler guarantees the member satisfies alignof(Containee) because the
// member type carries the alignas specifier; it inserts padding before m_data
// as needed.
class Container {
private:
   using pair_t = std::pair<int, Containee>;
   using coll_t = std::map<int, Containee>;
   using nested_coll_t = std::map<int, Wrapper>;

   char m_misalign;  // dummy member to show the compiler inserts padding
   Containee m_data; // embedded – no manual aligned allocation
   // pair_t       m_pair_data; // check on the run-time dictionary generated for pairs
   coll_t m_collection; // check on the run-time dictionary generated for collections
   nested_coll_t m_nested_collection;

public:
   Container() = default;
   explicit Container(int id) : m_misalign(0), m_data(id)
   {
      m_collection.emplace(id, Containee(id + 1));
      m_nested_collection.emplace(id, Wrapper(id + 2));
   }
   ~Container() = default; // embedded object is destroyed automatically

   // Demonstrate and verify alignment of the embedded element
   void showAlignment() const
   {
      const void *addr = static_cast<const void *>(&m_data);
      uintptr_t v = reinterpret_cast<uintptr_t>(addr);
      std::cout << "Containee alignment requirement: " << alignof(Containee) << '\n';
      std::cout << "m_data at " << addr << "  (addr % align = " << (v % alignof(Containee)) << ")\n";
      // runtime check
      assert((v % alignof(Containee)) == 0 && "m_data not correctly aligned");
   }

   Containee &get() { return m_data; }
   const Containee &get() const { return m_data; }

   ClassDef(Container, 2) // Container with embedded Containee
};

// AlternateContainer: same layout and behaviour as Container.
class AlternateContainer {
private:
   double padding;       // dummy member to show the compiler inserts padding
   char m_misalign;      // dummy member to show the compiler inserts padding
   Containee m_alt_data; // embedded – no manual aligned allocation

public:
   AlternateContainer() = default;
   explicit AlternateContainer(int id) : m_misalign(0), m_alt_data(id) {}
   ~AlternateContainer() = default; // embedded object is destroyed automatically

   // Demonstrate and verify alignment of the embedded element
   void showAlignment() const
   {
      const void *addr = static_cast<const void *>(&m_alt_data);
      uintptr_t v = reinterpret_cast<uintptr_t>(addr);
      std::cout << "Containee alignment requirement: " << alignof(Containee) << '\n';
      std::cout << "m_alt_data at " << addr << "  (addr % align = " << (v % alignof(Containee)) << ")\n";
      // runtime check
      assert((v % alignof(Containee)) == 0 && "m_data not correctly aligned");
   }

   void copyContainee(const Containee &c)
   {
      std::cout << "Copying Containee with id = " << c.id << " into AlternateContainer\n";
      // Guess the effective alignment of c's address: largest power-of-two
      // that divides the address.
      uintptr_t addr = reinterpret_cast<uintptr_t>(&c);
      uintptr_t guessed = 1;
      if (addr != 0) {
         guessed = addr & (~addr + 1); // isolate lowest set bit
      }
      std::cout << "  address of c: " << static_cast<const void *>(&c) << "  (guessed alignment: " << guessed
                << " bytes)\n";
      if (reinterpret_cast<uintptr_t>(&c) % alignof(Containee) != 0) {
         Error("copyContainer", "Containee object at %p does not satisfy alignment requirement of %zu\n", &c,
               alignof(Containee));
      }
      m_alt_data = c;
   }

   Containee &get() { return m_alt_data; }
   const Containee &get() const { return m_alt_data; }

   ClassDef(AlternateContainer, 2) // Alternate container with embedded Containee
};

#ifdef __ROOTCLING__
#pragma link C++ class Container + ;
#pragma link C++ class Containee + ;
#pragma link C++ class Wrapper + ;
#pragma link C++ class AlternateContainer + ;
#pragma read sourceClass = "Container" source = "char m_misalign" targetClass = "AlternateContainer" target = \
   "m_misalign" code = "{ m_misalign = onfile.m_misalign; }";
#pragma read sourceClass = "Container" source = "Containee m_data" targetClass = "AlternateContainer" target = \
   "m_alt_data" code = "{ newObj->copyContainee(onfile.m_data); }";
#endif

void writefile(const char *filename)
{
   Container c(42);
   c.showAlignment();
   std::cout << "c.get().id = " << c.get().id << '\n';

   TFile file(filename, "RECREATE");
   file.WriteObject(&c, "origContainer");
   file.Write();
};

void readfile(const char *filename)
{
   TFile file(filename, "READ");
   AlternateContainer *alt = file.Get<AlternateContainer>("origContainer");
   if (alt) {
      std::cout << "Successfully read AlternateContainer from file:\n";
      alt->showAlignment();
      std::cout << "Containee id = " << alt->get().id << '\n';
   } else {
      Fatal("readfile", "Failed to read AlternateContainer from file");
   }
};

int evolution()
{
   const char *filename = "alignment_evolution.root";
   writefile(filename);
   readfile(filename);
   return 0;
}

int main()
{
   return evolution();
}
