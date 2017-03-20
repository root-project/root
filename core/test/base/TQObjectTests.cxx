#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "Rtypes.h"
#include "TQObject.h"
#include "TQConnection.h"

#include "RQ_OBJECT.h"

// FIXME: We should think of a common place to put the mock objects.
class TQConnectionMock : public TQConnection {
public:
   virtual ~TQConnectionMock() {} // gmock requires it for cleanup on shutdown.
   MOCK_METHOD1(SetArg, void(Long_t param));
   MOCK_METHOD1(SetArg, void(ULong_t param));
   MOCK_METHOD1(SetArg, void(Float_t param));
   MOCK_METHOD1(SetArg, void(Double_t param));
   MOCK_METHOD1(SetArg, void(Long64_t param));
   MOCK_METHOD1(SetArg, void(ULong64_t param));
   MOCK_METHOD1(SetArg, void(const char *param));
   MOCK_METHOD2(SetArg, void(const Long_t *params, Int_t nparam /* = -1*/));

   MOCK_METHOD0(SendSignal, void());

   // MOCK_METHOD1(SendSignal, void());
};

#define Stringify(s) Stringifyx(s)
#define Stringifyx(s) #s

#define DICT_CLASS                                                           \
   class RQ_OBJECTTester : public TQObject {                                 \
      /* This will expand, adding signal/slot support to this class */       \
      RQ_OBJECT("RQ_OBJECTTester");                                          \
      Int_t fValue = 0;                                                      \
                                                                             \
   public:                                                                   \
      void SetValue(Int_t value)                                             \
      {                                                                      \
         /* to prevent infinite looping in the case of cyclic connections */ \
         if (value != fValue) {                                              \
            fValue = value;                                                  \
            Emit("SetValue(Int_t)", fValue);                                 \
         }                                                                   \
      }                                                                      \
      void PrintValue() const { printf("value=%d\n", fValue); }              \
      Int_t GetValue() const { return fValue; }                              \
   };

DICT_CLASS;

TEST(TQObject, Emit)
{
   gInterpreter->ProcessLine(Stringify(DICT_CLASS));
   RQ_OBJECTTester a;
   RQ_OBJECTTester b;
   a.Connect("SetValue(Int_t)", "RQ_OBJECTTester", &b, "SetValue(Int_t)");

   EXPECT_EQ(0, b.GetValue());

   a.SetValue(1);
   EXPECT_EQ(1, b.GetValue());
}
