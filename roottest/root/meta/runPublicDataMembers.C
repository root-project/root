
int Check(TList *l, const char *name)
{
   TObject *obj = l->FindObject(name);
   if (!obj) {
      std::cout << "Error: could not find " << name << '\n';
      return 1;
   }
   if (strcmp(obj->GetName(),name) != 0) {
      std::cout << "Error: Instead of " << name << " Found " << obj->GetName() << '\n';
      return 1;
   }
   return 0;
}

int CheckNotExist(TList *l, const char *name)
{
   TObject *obj = l->FindObject(name);
   if (obj) {
      if (strcmp(obj->GetName(),name) != 0) {
         std::cout << "Error: " << name << " actually exists" << '\n';
         return 1;
      }
      std::cout << "Error: Instead not finding " << name << " we found " << obj->GetName() << '\n';
      return 1;
   }
   return 0;
}

int runPublicDataMembers()
{
   TClass *cl = gROOT->GetClass("TNamed");
   TList *l = cl->GetListOfAllPublicDataMembers();

   int nFail = 0;
   nFail += Check(l, "kIsOnHeap");
   nFail += Check(l, "kZombie");
   nFail += CheckNotExist(l, "foo_bar");

   return nFail;
}
