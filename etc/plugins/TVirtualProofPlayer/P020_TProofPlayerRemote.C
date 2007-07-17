void P020_TProofPlayerRemote()
{
   gPluginMgr->AddHandler("TVirtualProofPlayer", "remote", "TProofPlayerRemote",
      "ProofPlayer", "TProofPlayerRemote(TProof*)");
}
