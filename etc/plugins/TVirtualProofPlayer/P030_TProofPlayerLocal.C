void P030_TProofPlayerLocal()
{
   gPluginMgr->AddHandler("TVirtualProofPlayer", "local", "TProofPlayerLocal",
      "ProofPlayer", "TProofPlayerLocal(TProof*)");
}
