int execenumSize()
{
   gSystem->Load("libenumSize");
   TEnum *e = TEnum::GetEnum("EOnlyShort");
   if (!e) {
      Error("execenumSize", "Could not find EOnlyShort");
      return 1;
   }
   auto dtype = e->GetUnderlyingType();
   if (dtype <= 0 || dtype >= kNumDataTypes) {
      Error("execenumSize", "No information available on the underlying type of EOnlyShort: %d", (int)dtype);
      return 2;
   }
   if (dtype != 2) {
      Error("execenumSize", "Incorrect information about the underlying type of EOnlyShort: %d", (int)dtype);
      return 3;
   }
   auto d = TDataType::GetDataType(dtype);
   if (!d) {
      Error("execenumSize", "Failed to retrieve the TDataType for EOnlyShort: %d", (int)dtype);
      return 4;
   }
   if (strcmp(d->GetName(),"short") != 0) {
      Error("execenumSize", "Incorrect information about the underlying type of EOnlyShort: %s", d->GetName());
      return 5;
   }
   return 0;
}
