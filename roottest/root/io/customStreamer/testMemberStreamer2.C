// This test is exactly the same as testMemberStreamer.C, except it doesn't have a statically-initialized
// custom streamer.
// Returns 0 on success, 1 on failure.
#include <TGenericClassInfo.h>
#include <TMemberStreamer.h>
#include <TBuffer.h>
#include <TClass.h>
#include <TFile.h>
#include <TError.h>
#include "MemberStreamerLib.h"

static int gTimesWritten, gTimesRead;

template <int DataValue>
void customMemberStreamer(TBuffer &R__b, void *objp, Int_t /*size*/)
{
   TClass *R__cl = TClass::GetClass("MemberStreamer");

   auto *obj = static_cast<MemberStreamerContainer *>(objp);
   UInt_t R__s, R__c;

   if (R__b.IsReading()) {
      /* Version_t v =  */ R__b.ReadVersion(&R__s, &R__c);

      R__b >> obj->fMember->fData;

      R__b.CheckByteCount(R__s, R__c, R__cl);

      ++gTimesRead;

   } else {
      R__c = R__b.WriteVersion(R__cl, kTRUE);

      obj->fMember->fData = DataValue;
      R__b << obj->fMember->fData;

      R__b.SetByteCount(R__c, kTRUE);

      ++gTimesWritten;
   }
}

namespace ROOT {
TGenericClassInfo *GenerateInitInstance(const MemberStreamerContainer *);
}

template <int DataValue>
bool setupMemberStreamer()
{
   ROOT::TGenericClassInfo *classInfo = ROOT::GenerateInitInstance((const MemberStreamerContainer *)nullptr);
   R__ASSERT(classInfo);
   return classInfo->AdoptMemberStreamer("fMember", new TMemberStreamer(customMemberStreamer<DataValue>));
}

static bool VerifyTimesWrittenAndRead(int expectedTimesWritten, int expectedTimesRead)
{
   if (gTimesWritten != expectedTimesWritten) {
      Error("checkMemberStreamer", "should have written %d times but actually written %d times!",
            expectedTimesWritten, gTimesWritten);
      return false;
   }
   if (gTimesRead != expectedTimesRead) {
      Error("checkMemberStreamer", "should have read %d times but actually read %d times!",
            expectedTimesRead, gTimesRead);
      return false;
   }
   return true;
}

bool checkMemberStreamer(int expectedDataValue)
{
   static int timesFunctionWasRun = 0;
   
   {
      MemberStreamerContainer container{};
      TFile file("testMemberStreamer.root", "RECREATE");
      file.WriteObject(&container, "container");
      // custom streamer should have modified the data
      if (container.fMember->fData != expectedDataValue) {
         Error("checkMemberStreamer", "value was supposed to be %d but it's %d", expectedDataValue,
               container.fMember->fData);
         return false;
      }
      file.Close();

      if (!VerifyTimesWrittenAndRead(timesFunctionWasRun + 1, timesFunctionWasRun))
         return false;
   }

   TFile file("testMemberStreamer.root", "READ");
   auto *container = file.Get<MemberStreamerContainer>("container");
   if (!VerifyTimesWrittenAndRead(timesFunctionWasRun + 1, timesFunctionWasRun + 1))
      return false;

   ++timesFunctionWasRun;
   return container && container->fMember->fData == expectedDataValue;
}

int testMemberStreamer()
{  
   if (!setupMemberStreamer<99>())
      return 1; // failure

   return !checkMemberStreamer(99);
}
