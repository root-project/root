void execenumsInNamespaces2(){
std::cout << "Enum first defined in the interpreter and then loaded from protoclass\n";
gInterpreter->ProcessLine("namespace mynamespace2{enum en{};}");
gInterpreter->ProcessLine("enum enGlobal{};");

auto a = TClass::GetClass("mynamespace2")->GetListOfEnums()->GetSize();
auto localList = dynamic_cast<TListOfEnums*>(gROOT->GetListOfEnums());
localList->Load();
auto aLocal = gROOT->GetListOfEnums()->GetSize();

gSystem->Load("libenumsInNamespaces2_dictrflx");

auto b = TClass::GetClass("mynamespace2")->GetListOfEnums()->GetSize();
auto bLocal = gROOT->GetListOfEnums()->GetSize();

std::cout << "Size of list of enums before loading the library: " << a << ", after " << b << std::endl;
if (b != 1){
   std::cerr << "Wrong number of enums: " << b << std::endl;
   TClass::GetClass("mynamespace2")->GetListOfEnums()->Dump();
   }

// Here we could have a different number of enums according to the installation, let's count them
bool enFound=false;
for (auto en : *gROOT->GetListOfEnums()){
  if (0 == strcmp(en->GetName(),"enGlobal")) {
     if (!enFound) enFound=true;
     else {
        cerr << "Two copies of enum enGlobal found!\n";
        return ;
        }
     }
}

}

