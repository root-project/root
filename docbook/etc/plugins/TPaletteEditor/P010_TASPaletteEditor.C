void P010_TASPaletteEditor()
{
   gPluginMgr->AddHandler("TPaletteEditor", "*", "TASPaletteEditor",
      "ASImageGui", "TASPaletteEditor(TAttImage*,UInt_t,UInt_t)");
}
