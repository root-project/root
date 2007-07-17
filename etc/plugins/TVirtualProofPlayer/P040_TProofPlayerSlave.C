void P040_TProofPlayerSlave()
{
   gPluginMgr->AddHandler("TVirtualProofPlayer", "slave", "TProofPlayerSlave",
      "ProofPlayer", "TProofPlayerSlave(TSocket*)");
}
