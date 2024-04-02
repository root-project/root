bool testing(TClass *cl, bool expected_value)
{
   if (expected_value == cl->HasCustomStreamerMember() ) {
      return true;
   } else {
      fprintf(stdout,"For %s we found %d rather than %d\n",cl->GetName(),cl->HasCustomStreamerMember(),expected_value);
      return false;
   }
}

int execProperties()
{
   bool result = testing(TObject::Class(),1);
   result &= testing(TNamed::Class(),0);
   result &= testing(TTree::Class(),1);
   result &= testing(TEntryList::Class(),1);
   result &= testing(TEntryListArray::Class(),0);  
   result &= testing(TString::Class(),1);

   // zero indicates succes;
   return (!result);
}
