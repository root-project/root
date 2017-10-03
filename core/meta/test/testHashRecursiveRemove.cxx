#include "TClass.h"
#include "TInterpreter.h"

#include "gtest/gtest.h"

const char* gCode = R"CODE(

   #include "TROOT.h"
   #include <iostream>

   void CallRecursiveRemoveIfNeeded(TObject &obj)
   {
      //std::cerr << "CallRecursiveRemoveIfNeeded: " << (void*)&obj << '\n';
      if (obj.TestBit(kMustCleanup)) {
         TROOT *root = ROOT::Internal::gROOTLocal;
         if (root) {
            if (root->MustClean()) {
               if (root == &obj)
                  return;
               //std::cerr << "CallRecursiveRemoveIfNeeded Inside: " << (void*)&obj << '\n';
               //root->GetListOfCleanups()->ls();
               root->GetListOfCleanups()->RecursiveRemove(&obj);
               // root->RecursiveRemove(&obj);
               obj.ResetBit(kMustCleanup);
            }
         }
      }
   }

   class FirstOverload : public TObject
   {
   public:
      virtual ULong_t     Hash() const { return 1; }

      ClassDef(FirstOverload, 2);
   };

   class SecondOverload : public FirstOverload // Could also have used TNamed.
   {
   public:
      virtual ULong_t     Hash() const { return 2; }

      ClassDef(SecondOverload, 2);
   };

   class SecondNoHash : public FirstOverload // Could also have used TNamed.
   {
   public:

      ClassDef(SecondNoHash, 2);
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

      ClassDefOverride(Third, 2);
   };

   class FirstOverloadCorrect : public TObject
   {
   public:
      ~FirstOverloadCorrect() {
         CallRecursiveRemoveIfNeeded(*this);
      }
      virtual ULong_t     Hash() const { return 3; }

      ClassDef(FirstOverloadCorrect, 2);
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
         CallRecursiveRemoveIfNeeded(*this);
      }

      virtual ULong_t Hash() const { return 4; }
      virtual int     Get() = 0;

      ClassDef(SecondCorrectAbstractHash, 2);
   };

   class ThirdCorrect : public SecondCorrectAbstract
   {
   public:
      int Get() override { return 0 ; };

      ClassDefOverride(ThirdCorrect, 2);
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

      ClassDefOverride(ThirdInCorrect, 2);
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

/*
virtual ULong_t     Hash() const;

RooSetPair
RooLinkedList
TStatsFeedback
TDrawFeedback
TVolume

*/