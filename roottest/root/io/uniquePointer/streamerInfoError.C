int streamerInfoError()
{
   B b;
   TFile f("testfile.root","RECREATE");
   f.WriteObjectAny(&b, "B", "b");
   f.Close();
   return 0; // Yes, we are checking errors elsewhere :)
}