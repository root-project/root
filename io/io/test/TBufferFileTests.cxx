#include "gtest/gtest.h"

#include "TBufferFile.h"
#include "TClass.h"
#include "TClonesArray.h"
#include "TObjString.h"
#include <vector>
#include <iostream>

// Tests ROOT-8367
TEST(TBufferFile, ROOT_8367)
{
   std::vector<int> v{1, 2, 3, 4, 5, 6, 7};

   TClass *cl1 = TClass::GetClass("vector<int>");
   TClass *cl2 = TClass::GetClass("vector<float>");

   TBufferFile *m_buff = new TBufferFile(TBuffer::kWrite);
   m_buff->WriteObjectAny(&v, cl1);
   m_buff->SetReadMode();
   m_buff->Reset();
   void *obj = m_buff->ReadObjectAny(cl2);
   std::vector<float> &v2 = *reinterpret_cast<std::vector<float> *>(obj);
   EXPECT_FLOAT_EQ(v2[0], 1.);
   EXPECT_FLOAT_EQ(v2[1], 2.);
   EXPECT_FLOAT_EQ(v2[2], 3.);
   EXPECT_FLOAT_EQ(v2[5], 6.);
   EXPECT_FLOAT_EQ(v2[6], 7.);
   EXPECT_EQ(v2.size(), 7);
}

// ROOT-6788
TEST(TBufferFile, TClonesArrayEmptySlotsStreaming)
{
	TClonesArray clArray(TObjString::Class(), 100);
	for (Int_t i=0; i<28; i++) {
		new (clArray[i]) TObjString();
	}
	for (Int_t i=0; i<27; i++) {
		delete clArray.RemoveAt(i);
	}
	
   TBufferFile buf(TBuffer::kWrite, 10000);
   buf.WriteObject(&clArray);
   buf.SetReadMode();
   buf.Reset();
   auto obj = buf.ReadObject(TClonesArray::Class());
   auto &readClArray = *dynamic_cast<TClonesArray*>(obj);
   EXPECT_EQ(readClArray.GetEntries(), clArray.GetEntries());
}

