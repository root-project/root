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
   // Trigger autoload.
   edm::AJet a;

   // Check whether std::less became invalid:
   TInterpreter::EErrorCode err = TInterpreter::kNoError;
   if (!gInterpreter->Declare(code))
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
