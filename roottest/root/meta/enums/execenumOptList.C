void execenumOptList()
{
   const char *tname = "TAttImage::EImageQuality";

   printf("List of values in enum %s\n", tname);
   if (auto enumDict = TEnum::GetEnum(tname)) {
      TIter iEnumConst(enumDict->GetConstants());
      while (auto enumConst = static_cast<TEnumConstant*>(iEnumConst())) {
        printf("    name %s   value %lld\n", enumConst->GetName(), enumConst->GetValue());
      }
   }

   auto cl = TClass::GetClass("TAttImage");
   auto member = cl->GetDataMember("fImageQuality");

   printf("List of options for TAttImage::fImageQuality member\n");
   TIter nxt(member->GetOptions());
   while (auto it = dynamic_cast<TOptionListItem*>(nxt())) {
      printf("    name %s label %s value %ld\n", it->fOptName.Data(), it->fOptLabel.Data(), it->fValue);
   }
}
