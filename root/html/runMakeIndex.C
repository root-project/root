void runMakeIndex() {
   THtml h;
   h.SetInputDir("$ROOTSYS");
   h.LoadAllLibs();
   h.MakeIndex();
}
