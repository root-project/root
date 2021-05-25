/// \file
/// \ingroup tutorial_vecops
/// \notebook -nodraw
/// In this tutorial we learn how the RVec class can be used to
/// adopt existing memory or allocate some.
///
/// \macro_code
/// \macro_output
///
/// \date May 2018
/// \author Danilo Piparo

// We use this class for didactic purposes: upon copy, a line is printed to the terminal.
class UponCopyPrinter {
public:
   UponCopyPrinter() = default;
   UponCopyPrinter(UponCopyPrinter &&) = default;
   UponCopyPrinter(const UponCopyPrinter &) { std::cout << "Invoking copy c'tor!" << std::endl; }
};

using namespace ROOT::VecOps;

void vo001_AdoptOrOwnMemory()
{

   // One of the essential features of RVec is its ability of adopting and owning memory.
   // Internally this is handled by the ROOT::Detail::VecOps::RAdoptAllocator class.

   // Let's create an RVec of UponCopyPrinter instances. We expect no printout:
   RVec<UponCopyPrinter> v(3);

   // Let's adopt the memory from v into v2. We expect no printout:
   RVec<UponCopyPrinter> v2(v.data(), v.size());

   // OK, let's check the addresses of the memory associated to the two RVecs It is the same!
   std::cout << v.data() << " and " << v2.data() << std::endl;

   // Now, upon reallocation, the RVec stops adopting the memory and starts owning it. And yes,
   // a copy is triggered. Indeed internally the storage of the RVec is an std::vector. Moreover,
   // the interface of the RVec is very, very similar to the one of std::vector: you have already
   // noticed it when the `data()` method was invoked, right?

   v2.push_back(UponCopyPrinter());

   // Of course, now the addresses are different.
   std::cout << v.data() << " and " << v2.data() << std::endl;
}
