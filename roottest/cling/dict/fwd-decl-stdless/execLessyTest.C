// CMS saw an issue with loading dictionaries that fwd declare std::less
// (because of specializations that their code provides). Because Sema sees
// re-decls of the default template arguments in these fwd decls it marks the
// most recent redecl of std::less as invalid.
// Subsequent use of std::less - in their case as a template template argument -
// causes errors.
//
// This test is producing such a dictionary and using std::less after parsing
// the fwd decls from the dictionary, as template template argument of UseLess.
// If std::less is marked invalid, the return value of
//   gInterpreter->ProcessLine("useless.get()")
// will be 0.

const char* code = R"CODE(
template <int I, template <class> class L>
struct UseLess{
   int get() {
      return L<edm::AJet>()(edm::AJet{12.}, edm::AJet{13.}) ? I : 17;
   }
};
UseLess<1001, std::less> useless;)CODE";

int execLessyTest()
{
   // Check whether std::less became invalid:
   TInterpreter::EErrorCode err = TInterpreter::kNoError;
   gInterpreter->ProcessLine(code, &err);
   if (err != TInterpreter::kNoError)
      return 1;

   long res = gInterpreter->ProcessLine("useless.get()", &err);
   if (err != TInterpreter::kNoError)
      return 2;

   if (res != 1001) {
      printf("RES=%ld\n", res);
      return 3;
   }

   return 0;
}
