void P030_TGWin32()
{
   gPluginMgr->AddHandler("TVirtualX", "win32", "TGWin32",
      "Win32", "TGWin32(const char*,const char*)");
   gPluginMgr->AddHandler("TVirtualX", "win32gdk", "TGWin32",
      "Win32gdk", "TGWin32(const char*,const char*)");
}
