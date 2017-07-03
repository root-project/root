int runf02(bool verbose = true) {
  // Test ROOT-6345: dynamic scope must repair if statements.
  gROOT->ProcessLine("if (true) UNKNOWN(42)"); // should complain about UNKNOWN
  // Test ROOT-7718: dynamic scope must repair array subscripts.
  gROOT->ProcessLine("ThisReallyDoesNotExist[42]"); // should complain about ThisReallyDoesNotExist
  return 0;
}
