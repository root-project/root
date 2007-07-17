void P010_TProofPlayer()
{
   gPluginMgr->AddHandler("TVirtualProofPlayer", "base", "TProofPlayer",
      "ProofPlayer", "TProofPlayer(TProof*)");
}
