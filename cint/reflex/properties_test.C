#include "Reflex/Type.h"
#include "Cintex/Cintex.h"
#include "TClass.h"
#include "TDataMember.h"
#include "TList.h"

void properties_test() {
   using namespace ROOT::Reflex;
   using namespace ROOT::Cintex;

   Scope sWithProperties = Scope::ByName("WithProperties");
   Type tWithProperties = sWithProperties;
   assert(tWithProperties);
   PropertyList props = tWithProperties.Properties();
   RflxAssert(!props.HasProperty("Doesn't Exist"));
   RflxAssert(props.HasProperty("id"));
   RflxEqual(props.PropertyAsString("ClassID"),"MyID");
   RflxAssert(props.HasProperty("someProperty"));
   RflxEqual(props.PropertyAsString("someProperty"),"42");

   struct CommentedMember {
      const char *name, *comment;
   };

   CommentedMember memco[5] = {
      {"memWithComment", "" /* not a iocomment: " a comment"*/},
      {"memNoIO", "! no I/O"},
      {"memWithoutSplitting", "|| no splitting"},
      {"memPtrAlwaysValid", "-> this pointer always points"},
      {"memDouble32Range", "[-1,0,12] 12 bits from -1 to 0"}
   };

   Cintex::Enable();
   TClass* cl = TClass::GetClass("WithProperties");
   RflxAssertT("TClass object", cl);
   if (!cl) return;
   for (int i = 0; i < 5; ++i) {
      Member mem = sWithProperties.MemberByName(memco[i].name);
      RflxAssertT(string("Reflex data member ") + memco[i].name, mem);
      RflxEqualT(string("Reflex comment property ") + memco[i].name,
                 mem.Properties().PropertyAsString("comment"),
                 memco[i].comment);
      // check CINT comment:
      if (memco[i].comment && memco[i].comment[0]) {
         TDataMember* dm = (TDataMember*)cl->GetListOfDataMembers()->FindObject(memco[i].name);
         RflxAssertT(string("TDataMember ") + memco[i].name, dm);
         if (!dm) continue;
         RflxEqualT(string("TDataMember comment ") + memco[i].name,
                    string(dm->GetTitle()), memco[i].comment);
      }
   }
}
