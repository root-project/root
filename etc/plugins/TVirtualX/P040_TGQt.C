void P040_TGQt()
{
   gPluginMgr->AddHandler("TVirtualX", "qt", "TGQt",
      "GQt", "TGQt(const char*,const char*)");
}
