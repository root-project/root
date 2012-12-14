void check(const char*arg) {
  tf = new TTreeFormula("test",arg,tree);
  tree->GetEntry(0);
  o = (TObject*)tf->EvalObject();
  o = (TObject*)tf->EvalObject();
  if (o) o->IsA()->Print();
  if (tf->EvalClass())
     tf->EvalClass()->Print();
}
