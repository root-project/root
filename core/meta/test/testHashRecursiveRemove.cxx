#include "TClass.h"
#include "TInterpreter.h"

#include "gtest/gtest.h"

const char* gCode = R"CODE(

   #include "TROOT.h"
   #include <iostream>

   class FirstOverload : public TObject
   {
   public:
      virtual ULong_t     Hash() const { return 1; }

      ClassDefInline(FirstOverload, 2);
   };

   class SecondOverload : public FirstOverload // Could also have used TNamed.
   {
   public:
      virtual ULong_t     Hash() const { return 2; }

      ClassDefInline(SecondOverload, 2);
   };

   class SecondNoHash : public FirstOverload // Could also have used TNamed.
   {
   public:

      ClassDefInline(SecondNoHash, 2);
   };

   class SecondAbstract : public FirstOverload // Could also have used TNamed.
   {
   public:
      virtual int Get() = 0;

      ClassDef(SecondAbstract, 2);
   };

   class Third : public SecondAbstract
   {
   public:
      int Get() override { return 0 ; };

      ClassDefInlineOverride(Third, 2);
   };

   class FirstOverloadCorrect : public TObject
   {
   public:
      ~FirstOverloadCorrect() {
         ROOT::CallRecursiveRemoveIfNeeded(*this);
      }
      virtual ULong_t     Hash() const { return 3; }

      ClassDefInline(FirstOverloadCorrect, 2);
   };

   class SecondCorrectAbstract : public FirstOverloadCorrect // Could also have used TNamed.
   {
   public:
      virtual int Get() = 0;

      ClassDef(SecondCorrectAbstract, 2);
   };

   class SecondCorrectAbstractHash : public FirstOverloadCorrect // Could also have used TNamed.
   {
   public:
      ~SecondCorrectAbstractHash() {
         ROOT::CallRecursiveRemoveIfNeeded(*this);
      }

      virtual ULong_t Hash() const { return 4; }
      virtual int     Get() = 0;

      ClassDef(SecondCorrectAbstractHash, 2);
   };

   class ThirdCorrect : public SecondCorrectAbstract
   {
   public:
      int Get() override { return 0 ; };

      ClassDefInlineOverride(ThirdCorrect, 2);
   };

   class SecondInCorrectAbstract : public FirstOverloadCorrect // Could also have used TNamed.
   {
   public:
      virtual ULong_t Hash() const { return 4; }
      virtual int     Get() = 0;

      ClassDef(SecondInCorrectAbstract, 2);
   };

   class ThirdInCorrect : public SecondInCorrectAbstract
   {
   public:
      int Get() override { return 0 ; };

      ClassDefInlineOverride(ThirdInCorrect, 2);
   };

)CODE";

const char *gErrorOutput = R"OUTPUT(Error in <ROOT::Internal::TCheckHashRecurveRemoveConsistency::CheckRecursiveRemove>: The class FirstOverload overrides TObject::Hash but does not call TROOT::RecursiveRemove in its destructor.
Error in <ROOT::Internal::TCheckHashRecurveRemoveConsistency::CheckRecursiveRemove>: The class SecondOverload overrides TObject::Hash but does not call TROOT::RecursiveRemove in its destructor.
Error in <ROOT::Internal::TCheckHashRecurveRemoveConsistency::CheckRecursiveRemove>: The class FirstOverload overrides TObject::Hash but does not call TROOT::RecursiveRemove in its destructor.
Error in <ROOT::Internal::TCheckHashRecurveRemoveConsistency::CheckRecursiveRemove>: The class FirstOverload overrides TObject::Hash but does not call TROOT::RecursiveRemove in its destructor.
Error in <ROOT::Internal::TCheckHashRecurveRemoveConsistency::CheckRecursiveRemove>: The class SecondInCorrectAbstract overrides TObject::Hash but does not call TROOT::RecursiveRemove in its destructor.
)OUTPUT";


void DeclareFailingClasses() {
   gInterpreter->Declare(gCode);
}

TEST(HashRecursiveRemove,RootClasses)
{
   EXPECT_TRUE(TClass::GetClass("TObject")->HasConsistentHashMember());
   EXPECT_TRUE(TClass::GetClass("TNamed")->HasConsistentHashMember());
   EXPECT_FALSE(TClass::GetClass("TH1")->HasConsistentHashMember());
   EXPECT_TRUE(TClass::GetClass("TH1F")->HasConsistentHashMember());
}

TEST(HashRecursiveRemove,FailingClasses)
{
   DeclareFailingClasses();

   testing::internal::CaptureStderr();

   EXPECT_NE(nullptr,TClass::GetClass("FirstOverload"));
   EXPECT_FALSE(TClass::GetClass("FirstOverload")->HasConsistentHashMember());
   EXPECT_FALSE(TClass::GetClass("SecondOverload")->HasConsistentHashMember());
   EXPECT_FALSE(TClass::GetClass("SecondNoHash")->HasConsistentHashMember());

   EXPECT_FALSE(TClass::GetClass("SecondAbstract")->HasConsistentHashMember());
   EXPECT_FALSE(TClass::GetClass("Third")->HasConsistentHashMember());
   EXPECT_TRUE(TClass::GetClass("FirstOverloadCorrect")->HasConsistentHashMember());

   EXPECT_FALSE(TClass::GetClass("SecondCorrectAbstract")->HasConsistentHashMember());
   EXPECT_FALSE(TClass::GetClass("SecondCorrectAbstractHash")->HasConsistentHashMember());
   EXPECT_TRUE(TClass::GetClass("ThirdCorrect")->HasConsistentHashMember());
   EXPECT_FALSE(TClass::GetClass("SecondInCorrectAbstract")->HasConsistentHashMember());
   EXPECT_FALSE(TClass::GetClass("ThirdInCorrect")->HasConsistentHashMember());

   std::string output = testing::internal::GetCapturedStderr();
   EXPECT_EQ(gErrorOutput,output);
}
