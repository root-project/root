void P010_TDataProgressDialog()
{
   gPluginMgr->AddHandler("TDataProgressDialog", "*", "TDataProgressDialog",
      "PeacGui", "TDataProgressDialog(TProof*,const char*,Int_t,Long64_t)");
}
