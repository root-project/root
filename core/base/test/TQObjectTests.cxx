#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "Rtypes.h"
#include "TQObject.h"
#include "TQConnection.h"

#include "RQ_OBJECT.h"

#define Stringify(s) Stringifyx(s)
#define Stringifyx(s) #s

// The interpreter needs to know about RQ_OBJECTTester and using this trick avoids moving this non-reusable class into
// its own header file.
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
