// Autoparsing a payload from within a template instantiation used to crash in
// Sema::EnterTemplatedContext, because the nested parse saw the function scopes
// of the outer instantiation (Sema::FunctionScopesStart was not reset).
int execNestedAutoParse()
{
   // Registers only the forward declaration of Payload; its definition is
   // parsed on demand.
   if (gSystem->Load("libnestedAutoParse_dictrflx") < 0)
      return 1;
   // Use ProcessLine rather than TInterpreter::Declare: Declare suspends
   // autoparsing.
   if (gROOT->ProcessLine(".L useNestedAutoParse.C") != 0)
      return 2;
   return gROOT->ProcessLine("useNestedAutoParse();") == 42 ? 0 : 3;
}
