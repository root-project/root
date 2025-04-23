void MakeNameCyclesRootmvInput()
{
   TFile f("nameCyclesRootmvInput.root", "recreate");
   TTree t("tree", "tree");
   int x = 42;
   t.Branch("x", &x);
   t.Fill();
   t.Write(); // first namecycle
   t.Fill();
   t.Write(); // second namecycle
   f.Close();
}
