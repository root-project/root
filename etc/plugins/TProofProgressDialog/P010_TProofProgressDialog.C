void P010_TProofProgressDialog()
{
   gPluginMgr->AddHandler("TProofProgressDialog", "*", "TProofProgressDialog",
      "SessionViewer", "TProofProgressDialog(TProof*,const char*,Int_t,Long64_t,Long64_t)");
}
