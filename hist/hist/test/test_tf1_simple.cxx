#include "TBufferFile.h"
#include "TClass.h"
#include "TFitResult.h"
#include "TH1F.h"
#include "TF1.h"

#include "gtest/gtest.h"

#include <iostream>

template<typename T>
std::unique_ptr<T> SerialiseDeserialise(const T &f)
{
   // code inspired from TDirectroy::CloneObject
   // works only for TObjects

   auto  pobj       = (char *)f.IsA()->New();
   Int_t baseOffset = f.IsA()->GetBaseClassOffset(TObject::Class());
   auto  newobj     = (TF1 *)(pobj + baseOffset);

   auto        fptr = &f;
   TBufferFile buffer(TBuffer::kWrite, 10000);
   ((T *)fptr)->Streamer(buffer);

   // read new object from buffer
   buffer.SetReadMode();
   buffer.ResetMap();
   buffer.SetBufferOffset(0);
   newobj->Streamer(buffer);
   newobj->ResetBit(kIsReferenced);
   newobj->ResetBit(kCanDelete);

   return std::unique_ptr<T>(newobj);
}

// issue #16184
TEST(TF1, PersistentNumber)
{
   TH1F h("myHist", "myTitle", 64, -4, 4);
   h.FillRandom("gaus");
   TF1 f1("f1", "gaus");
   auto f1_d = SerialiseDeserialise(f1);

   auto res = h.Fit(&f1, "SQN");
   EXPECT_EQ(0, res->Status());

   res = h.Fit(f1_d.get(), "SQN");
   EXPECT_EQ(0, res->Status());
}