// Broken recursion in TFormula.
{
  // Create 1st level function.

  TF1 f1("f1", "[0]");

  // Set 1st level parameter.

  f1.SetParameter(0, 1);

  // Create 2nd level function.

  TF1 f2("f2", "[0]+f1");

  // Set 2nd level parameter.

  f2.SetParameter(0, 2);

  // Inspect 2nd level function. 1st level parameter has offset 1 in
  // parameter array and expression array.

  f2.Print();

  // Therefore, 2nd level function evaluates properly to 3.

  f2.Eval(0);

  // Create 3rd level function.

  TF1 f3("f3", "[0]+[1]+f2");

  // Set 3rd level parameter.

  f3.SetParameter(0, 3);

  // Inspect 3rd level function. 1st level parameter has offset 3 in
  // parameter array, but offset 2 in expression array.

  f3.Print();

  // Therefore, 3rd level function with proper value 6 evaluates
  // improperly to 7.

  f3.Eval(0);

#ifdef ClingWorkAroundBrokenUnnamedReturn
  int res = 0;
#else
  return 0;
#endif
}

