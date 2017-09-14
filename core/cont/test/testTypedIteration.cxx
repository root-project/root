#include "TCollection.h"

#include "TBaseClass.h"
#include "TClass.h"
#include "TList.h"
#include "TNamed.h"

#include "gtest/gtest.h"

const char *gCode = R"CODE(
   struct ClassWithOverlap : public TObject {
      enum EStatusBits {
         kOverlappingBit = BIT(13)
      };
   };
)CODE";

TEST(TCollection, RangeCast)
{
   using namespace ROOT::Detail;

   TClass *cl = TNamed::Class();
   TClass *tobjectCl = TObject::Class();
   TClass *baseClassCl = TBaseClass::Class();

   for (auto bcl : *cl->GetListOfBases()) {
      // bcl is actually a TObject*
      EXPECT_EQ(tobjectCl, bcl->Class());
   }

   for (auto bcl : TRangeStaticCast<TBaseClass>(*cl->GetListOfBases())) {
      // bcl is actually a TBaseClass*
      EXPECT_EQ(baseClassCl, bcl->Class());
   }

   for (auto bcl : TRangeStaticCast<TBaseClass>(cl->GetListOfBases())) {
      // bcl is actually a TBaseClass*
      EXPECT_EQ(baseClassCl, bcl->Class());
   }

   for (auto bcl : TRangeDynCast<TClass>(*cl->GetListOfBases())) {
      // bcl is actually a TBaseClass*
      EXPECT_EQ(bcl, nullptr);
   }

   for (auto bcl : TRangeDynCast<TClass>(cl->GetListOfBases())) {
      // bcl is actually a TBaseClass*
      EXPECT_EQ(bcl, nullptr);
   }
}

TEST(TCollection, TypedIter)
{
   using namespace ROOT::Detail;

   TClass *cl = TNamed::Class();
   TClass *tobjectCl = TObject::Class();
   TClass *baseClassCl = TBaseClass::Class();

   TIter next(cl->GetListOfBases());
   while (auto bcl = next()) {
      // bcl is actually a TObject*
      EXPECT_EQ(tobjectCl, bcl->Class());
   }

   TTypedIter<TBaseClass> bnext(cl->GetListOfBases());
   while (auto bcl = bnext()) {
      // bcl is actually a TBaseClass*
      EXPECT_EQ(baseClassCl, bcl->Class());
   }

   ROOT::Internal::TRangeDynCastIterator<TClass> dynnext(cl->GetListOfBases());
   while (dynnext.Next() || dynnext != cl->GetListOfBases()->end()) {
      auto bcl = *next;
      EXPECT_EQ(bcl, nullptr);
   }

/// Test compilation failure
#if 0
   TDynTypedIter<TBaseClass> dynnextfail(cl->GetListOfBases());
   while(auto bcl = dynnextfail()) {
      // bcl is actually a TBaseClass*
      EXPECT_EQ(baseClassCl, bcl->Class());
   }
#endif
}
