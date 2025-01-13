class Ringer {};

class FailPrinter
{
  bool fEnabled = true;
  TClass *fClass = nullptr;
public:
  FailPrinter(TClass *cl) : fClass(cl) {};
  ~FailPrinter() {
    if (fEnabled && fClass)
      fClass->GetStreamerInfos()->ls();
  }
  void Clear() {
    fEnabled = false;
  }
};

int execAddRemoveStreamerInfo() {

  auto cl = TClass::GetClass("Ringer");
  if (!cl) {
    std::cout << "Failed to load the TClass for the class 'Ringer'\n";
    return 1;
  }

  FailPrinter failPrinter(cl);

  ROOT::ResetClassVersion(cl, "Ringer", 3);

  auto version = cl->GetClassVersion();
  if (version != 3) {
    std::cout << "Failed to set the version number, instead of 3 we got: " << version << '\n';
     return 2;
  }
  auto current_info = cl->GetStreamerInfo();
  if (!current_info) {
    std::cout << "Failed to load the current StreamerInfo\n";
    return 3;
  }
  auto other = (TStreamerInfo*)current_info->Clone();
  other->SetClassVersion(2);
  other->SetClass(cl);
  cl->RegisterStreamerInfo(other);
  // other->BuildOld();

  auto check = cl->GetStreamerInfo(3);
  if (!check) {
    std::cout << "No StreamerInfo in slot 3\n";
    return 4;
  }
  check = cl->GetStreamerInfo(2);
  if (!check) {
    std::cout << "No StreamerInfo in slot 2\n";
    return 5;
  }
  // Now remove a specific entry:
  cl->RemoveStreamerInfo(2);
  check = dynamic_cast<TStreamerInfo*>(cl->GetStreamerInfos()->At(2));
  if (check) {
    std::cout << "After removal there is still a StreamerInfo in slot 2 (" << check->GetClassVersion() << ")\n";
    return 6;
  }
  cl->RemoveStreamerInfo(3);
  check = dynamic_cast<TStreamerInfo*>(cl->GetStreamerInfos()->At(3));
  if (check) {
    std::cout << "After removal there is still a StreamerInfo in slot 3 (" << check->GetClassVersion() << ")\n";
    return 7;
  }
  failPrinter.Clear();
  return 0;
}


