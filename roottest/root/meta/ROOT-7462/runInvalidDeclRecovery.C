#include "invalidDeclRecovery.h"
#include "TInterpreter.h"

int runInvalidDeclRecovery()
{
   gInterpreter->Declare("edm::Wrapper<edm::SortedCollection<ESKCHIPBlock, edm::StrictWeakOrdering<ESKCHIPBlock> > > a;");
   gInterpreter->Declare("edm::SortedCollection<ESKCHIPBlock,edm::StrictWeakOrdering<ESKCHIPBlock> > b;");

   gInterpreter->Declare(R"foo(
                         class ESKCHIPBlock
                         {
                         public:
                            typedef int key_type;
                         };
                         )foo");

   gInterpreter->Declare("edm::Wrapper<edm::SortedCollection<ESKCHIPBlock, edm::StrictWeakOrdering<ESKCHIPBlock> > > c;");
   gInterpreter->Declare("edm::SortedCollection<ESKCHIPBlock,edm::StrictWeakOrdering<ESKCHIPBlock> > d;");

   printf("Success\n");
   return 0;
}
