void valPrinterTypeId()
{
   gInterpreter->ProcessLine("std::type_index(typeid(TObject)).name()");
   gInterpreter->ProcessLine("std::type_index(typeid(int)).name()");
}
