#define TEST_RUNTIME 1

void runAvoidParsing(bool space = true)
{
   if(space)
      TClass::GetClass("testing::UserClass<testing::FindUsingAdvance<testing::InnerContent> >");
   else
      TClass::GetClass("testing::UserClass<testing::FindUsingAdvance<testing::InnerContent>>");
}
