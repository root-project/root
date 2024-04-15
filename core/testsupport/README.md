# ROOT::TestSupport: the unit-test support library

This library supports ROOT's unit tests. It implements two main functions
1. It provides a static library target `ROOT::TestSupport`. All google tests that are defined using `ROOT_ADD_GTEST` will be linked against this target.
   When a test executable starts up, this will install a ROOT message handler that intercepts all messages / diagnostics.
   If a message with severity > kInfo is issued, this message handler will register a test failure.

   This way, we are ensuring that no gtest can issue unnoticed warning or error messages.
2. However, some warnings and errors are expected as the result of certain tests. Therefore, the library provides tools to declare when messages are expected during a test. For this,
   1. Include the header `ROOT/TestSupport.hxx`.
   2. Declare a RAII object that temporarily replaces the message handler from 1.
   3. Register the expected messages to this object, so it can check that they are indeed sent.

   This could look as follows:
   ```c++
   #include <ROOT/TestSupport.hxx>

   // In a test function:
   ROOT::TestSupport::CheckDiagsRAII checkDiag;
   checkDiag.requiredDiag(kError, "prepareMethod", "Can't compile function TFormula", /*matchFullMessage=*/false);
   checkDiag.requiredDiag(kError, "TFormula::InputFormulaIntoCling", "Error compiling formula expression in Cling", true);
   // run test that generates the above errors
   ```
