void P060_TProofPlayerLite()
{
   gPluginMgr->AddHandler("TVirtualProofPlayer", "lite", "TProofPlayerLite",
      "ProofPlayer", "TProofPlayerLite(TProof*)");
}
