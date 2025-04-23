int MissingInclude() {
  TInterpreter::EErrorCode error = TInterpreter::kRecoverable;
  gInterpreter->ProcessLine("#include \"TObject.h\"", &error);
  if (error != TInterpreter::kNoError) {
    std::cerr << "Found include but error code is " << error << '\n';
    exit(1);
  }
  gInterpreter->ProcessLine("#include \"This/File/Does/Not/Exist.Please\"", &error);
  if (error != TInterpreter::kRecoverable) {
    std::cerr << "Expected \"recoverable\" error due to missing include, but error code is " << error << '\n';
    exit(1);
  }
  // Trigger autoparsing:
  gInterpreter->ProcessLine("#include \"TriggerAutoParse.h\"", &error);
  if (error != TInterpreter::kRecoverable) {
    std::cerr << "Expected \"recoverable\" error due to missing include, but error code is " << error << '\n';
    exit(1);
  }
  return 0;
}
