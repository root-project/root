## \file
## \ingroup tutorial_vecops
## \notebook -nodraw
## In this tutorial we learn how the RVec class can be used to
## adopt existing memory or allocate some.
##
## \macro_code
## \macro_output
##
## \date May 2018
## \author Danilo Piparo

import ROOT

# We use this class for didactic purposes: upon copy, a line is printed to the terminal.

ROOT.gInterpreter.Declare('''
class UponCopyPrinter {
public:
   UponCopyPrinter() = default;
   UponCopyPrinter(UponCopyPrinter &&) = default;
   UponCopyPrinter(const UponCopyPrinter &) { std::cout << "Invoking copy c'tor!" << std::endl; }
};
''')

RVec_UponCopyPrinter = ROOT.ROOT.VecOps.RVec(ROOT.UponCopyPrinter)

# One of the essential features of RVec is its ability of adopting and owning memory.

# Let's create an RVec of UponCopyPrinter instances. We expect no printout:
v = RVec_UponCopyPrinter(3)

# Let's adopt the memory from v into v2. We expect no printout:
v2 = RVec_UponCopyPrinter(v.data(), v.size())

# OK, let's check the addresses of the memory associated to the two RVecs It is the same!
print("%s and %s" %(v.data(), v2.data()))

# Now, upon reallocation, the RVec stops adopting the memory and starts owning it. And yes,
# a copy is triggered. Indeed internally the storage of the RVec is an std::vector. Moreover,
# the interface of the RVec is very, very similar to the one of std::vector: you have already
# noticed it when the `data()` method was invoked, right?
v2.resize(4)

# Of course, now the addresses are different.
print("%s and %s" %(v.data(), v2.data()))
