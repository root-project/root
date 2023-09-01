void P050_TProofPlayerSuperMaster()
{
   gPluginMgr->AddHandler("TVirtualProofPlayer", "sm", "TProofPlayerSuperMaster",
      "ProofPlayer", "TProofPlayerSuperMaster(TProof*)");
}
