// Test ROOT-6345: dynamic scope must repair if statements.
void runf02(bool verbose = true) {
  if (verbose) UNKNOWN("max\n"); // should complain about UNKNOWN
}
