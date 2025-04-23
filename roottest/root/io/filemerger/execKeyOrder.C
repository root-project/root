int execKeyOrder()
{
   TFile f("inputKeyOrder.root","recreate");

   TNamed alt("alt","");

   alt.Write("Alt_01");
   TNamed obj("first","first");

   obj.Write();

   obj.SetTitle("Second");
   obj.Write();

   obj.SetTitle("Third");
   obj.Write();

   alt.Write("Alt_02");

   obj.SetTitle("Fourth");
   obj.Write();

   alt.Write("Alt_03");

   f.ls();
   f.Close();

   TFileMerger fileMerger(kFALSE, kFALSE);
   fileMerger.SetMsgPrefix("testKeyOrder");
   fileMerger.AddFile("inputKeyOrder.root");
   fileMerger.SetPrintLevel(-1);
   fileMerger.OutputFile("outputKeyOrder.root");

   fileMerger.PartialMerge();

   TFile o("outputKeyOrder.root");
   o.ls();


   return 0;
}
