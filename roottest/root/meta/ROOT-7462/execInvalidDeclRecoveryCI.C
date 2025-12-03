#include "invalidDeclRecovery.h"
#include "TInterpreter.h"

void testInvalidDecl() {

   // (a) gCling->ClassInfo_Factory("edm::SortedCollection<ESKCHIPBlock,edm::StrictWeakOrdering<ESKCHIPBlock> >")

   // (b) gCling->ClassInfo_Factory("edm::Wrapper<edm::SortedCollection<ESKCHIPBlock, edm::StrictWeakOrdering<ESKCHIPBlock> > >")
   // Issues compilation error which are not recovered from
   // i.e. leads to invalid decl. if done before (a).

   // The following 'works':
   //   gInterpreter->ClassInfo_Factory("edm::SortedCollection<ESKCHIPBlock,edm::StrictWeakOrdering<ESKCHIPBlock> >");
   //   gInterpreter->ClassInfo_Factory("edm::Wrapper<edm::SortedCollection<ESKCHIPBlock, edm::StrictWeakOrdering<ESKCHIPBlock> > >");
   //   gInterpreter->ClassInfo_Factory("edm::SortedCollection<ESKCHIPBlock,edm::StrictWeakOrdering<ESKCHIPBlock> >");

   // The following failsl:

   gInterpreter->ClassInfo_Factory("edm::Wrapper<edm::SortedCollection<ESKCHIPBlock, edm::StrictWeakOrdering<ESKCHIPBlock> > >");
   gInterpreter->ClassInfo_Factory("edm::SortedCollection<ESKCHIPBlock,edm::StrictWeakOrdering<ESKCHIPBlock> >");
}

void testValidDecl() {
   // In the real life example, this comes from autoparsing, but it is not
   // 'provokes' in the real use case by the code above as they are doing
   // during the 'TClass unload' (SetClassInfo which explicitly disable autoparsing).
   // [Happens to happen at the end of the process]
   gInterpreter->Declare(R"foo(
                         class ESKCHIPBlock
                         {
                         public:
                            typedef int key_type;
                         };
                         )foo");

   // (c) gCling->ClassInfo_Factory("edm::SortedCollection<ESKCHIPBlock,edm::StrictWeakOrdering<ESKCHIPBlock> >")
   // is fine here (if (b) not executed)

   gInterpreter->ClassInfo_Factory("edm::Wrapper<edm::SortedCollection<ESKCHIPBlock, edm::StrictWeakOrdering<ESKCHIPBlock> > >");
   gInterpreter->ClassInfo_Factory("edm::SortedCollection<ESKCHIPBlock,edm::StrictWeakOrdering<ESKCHIPBlock> >");
}


int execInvalidDeclRecoveryCI()
{
   testInvalidDecl();
   testValidDecl();

   //printf("End of test\n");
   return 0;
}
