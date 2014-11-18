// Test ROOT-6345: dynamic scope must repair if statements.
void runf02(bool verbose = true) {
  gROOT->ProcessLine("if (true) UNKNOWN(42)"); // should complain about UNKNOWN
}
