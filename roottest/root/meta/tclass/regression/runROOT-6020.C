{
   int result = 0;
   TClass *c1 = TClass::GetClass("basic_iostream<char,char_traits<char> >");
   TClass *c2 = TClass::GetClass("std::iostream");
   TClass *c3 = TClass::GetClass("basic_iostream<char>");
   if (c1 != c2) {
      ++result;
      Error("ROOT-6020","The TClass are not matching for std::iostream and basic_iostream<char,char_traits<char> >");
   }
   if (c2 != c3) {
      ++result;
      Error("ROOT-6020","The TClass are not matching for std::iostream and basic_iostream<char>");
   }

   #include <ostream>
   #include <sstream>

   TClass *c4 = TClass::GetClass("std::stringstream");
   if (!c4) {
      ++result;
      Error("ROOT-6020","Can not find std::stringstream");
   } else {
      TClass *b1 = c4->GetBaseClass("std::ostream");
      if (!b1) {
         ++result;
         Error("ROOT-6020","Can not find base class std::ostream");
      }
      TClass *b2 = c4->GetBaseClass("std::iostream");
      if (!b2) {
         ++result;
         Error("ROOT-6020","Can not find base class std::iostream");
      }
      TClass *b3 = c4->GetBaseClass("basic_iostream<char,char_traits<char> >");
      if (!b3) {
         ++result;
         Error("ROOT-6020","Can not find base class basic_iostream<char,char_traits<char> >");
      }
      if (b2 != b3) {
         Error("ROOT-6020","TClass are not matching for base class iostream");
      }
   }
   return result;
}
