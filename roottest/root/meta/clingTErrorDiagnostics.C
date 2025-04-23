namespace {
   void TestErrHdlr(int level, Bool_t abort, const char *location,
                    const char *msg)
   { fprintf(stderr, "%s: %d: %s\n", __func__, level, msg); }
}

void clingTErrorDiagnostics() {
   ::SetErrorHandler(TestErrHdlr);

   gInterpreter->ProcessLine("int f1() { return; }");

   gInterpreter->ReportDiagnosticsToErrorHandler();
   gInterpreter->ProcessLine("int f2() { return; }");

   gInterpreter->ReportDiagnosticsToErrorHandler(/*enable=*/false);
   // This should revert to regular cling diagnostics
   gInterpreter->ProcessLine("int f3() { return; }");
}
