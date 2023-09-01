void P010_TGPasswdDialog()
{
   gPluginMgr->AddHandler("TGPasswdDialog", "*", "TGPasswdDialog",
      "Gui", "TGPasswdDialog(const char*,char*,Int_t,UInt_t,UInt_t)");
}
