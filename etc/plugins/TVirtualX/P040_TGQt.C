TVirtualX* ROOT_Plugin_TGQt(const char*,const char*);

void P040_TGQt()
{
   gPluginMgr->AddHandler("TVirtualX", "qt", "TGQt",
      "GQt", "::ROOT_Plugin_TGQt(const char*,const char*)");
}
