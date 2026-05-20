{
gInterpreter->SetClassAutoLoading(false);
gSystem->Load("libConfigType");
auto c = TClass::GetClass("std::tuple<ConfigType, std::string>");
std::cout << "c is " << c << '\n';
std::cout << "c has State: " << c->GetState() << '\n';
gSystem->Unload("libConfigType");
auto c2 = TClass::GetClass("std::tuple<ConfigType, std::string>");
std::cout << "c2 is " << c2 << '\n';
std::cout << "c2 has State: " << c2->GetState() << '\n';                                                                                                 
return 0;

}
