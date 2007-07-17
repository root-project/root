void P010_TProofChain()
{
   gPluginMgr->AddHandler("TChain", "^proof", "TProofChain",
      "Proof", "TProofChain(TChain *)");
}
