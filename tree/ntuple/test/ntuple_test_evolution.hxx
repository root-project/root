#ifndef ROOT_RNTuple_Test_Evolution
#define ROOT_RNTuple_Test_Evolution

// Helper code for RNTuple schema evolution unit tests

namespace ROOT::Internal {

void EvaluateIntImpl(const char *expression, int *value)
{
   auto interpreterValue = gInterpreter->MakeInterpreterValue();
   ASSERT_TRUE(gInterpreter->Evaluate(expression, *interpreterValue));
   *value = interpreterValue->GetAsLong();
}

} // namespace ROOT::Internal

#define EXPECT_EVALUATE_EQ(expression, expected)            \
   do {                                                     \
      int _value;                                           \
      ROOT::Internal::EvaluateIntImpl(expression, &_value); \
      if (::testing::Test::HasFatalFailure())               \
         return;                                            \
      EXPECT_EQ(expected, _value);                          \
   } while (0)

#endif
