void LDAPExample()
{
   gSystem->Load("libRLDAP.so");

   TLDAPServer *server = new TLDAPServer("ldap.cern.ch");

   TLDAPResult *result = server.Search();

   if (result == 0) {
      printf("Search failed\n");
      exit(1);
   }
   result->Print();
   delete result;

   const char *namingcontexts = server.GetNamingContexts();
   result = server.Search(namingcontexts, LDAP_SCOPE_ONELEVEL, 0, 0, 1);
   TLDAPEntry *entry = result.GetNext();
   entry->Print();

   TString dn = entry->GetDn();

   delete result;
   delete entry;

   cout << "The DN of the entry is " << dn << endl;

   result = server.Search(dn, LDAP_SCOPE_SUBTREE, 0, 0, 0);

   if (result == 0) {
      printf("Search failed\n");
      exit(1);
   }

   result->Print();
   Int_t counter = result.GetCount();
   cout << "The result contains " << counter << " entries !!!" << endl;

   entry = result.GetNext();

   TLDAPAttribute *attribute = entry.GetAttribute("member");

   Int_t counter2 = attribute.GetCount();
   cout << "The attribute " << attribute.GetName() << " contains "
        << counter2 << " values !!!" << endl;
   const char *value = attribute.GetValue();
   cout << "The first value of the attribute is " << endl;
   cout << value << endl;

   delete result;
   delete entry;
}
