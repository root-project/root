int runDupe()
{
  TTree tree("testTree", "my test tree");
  ULong64_t run = 5;
  ULong64_t event = 1;
  tree.Branch("run", &run, "run/l");
  tree.Branch("event", &event, "event/l");
  tree.Fill();
  tree.Fill();

  tree.BuildIndex("run", "event");

  return 0;
}