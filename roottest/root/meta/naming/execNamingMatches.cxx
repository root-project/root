#include "TROOT.h"
#include "THashTable.h"
#include "TObjArray.h"
#include "TFile.h"

#include "TClass.h"
#include "TClassTable.h"
#include "TProtoClass.h"

#include "TError.h"

int CheckDictionary(TObjArray *proto, const char *longname, const char *shortname) {
   int failedtest = 0;

   if (! TClassTable::GetDictNorm(longname) ) {
      Error("CheckDictionary","Dictionary function not found for '%s'",longname);
      ++failedtest;
      if (TClassTable::GetDictNorm(shortname)) {
         Warning("CheckDictionary","But it was found for '%s'",shortname);
         ++failedtest;
      }
   }
   TObject *pClass = proto->FindObject(longname);
   if (!pClass) {
      Error("CheckDictionary","Proto class not found with the long name: %s",longname);
      ++failedtest;
   }
   pClass = proto->FindObject(shortname);
   if (pClass) {
      Error("CheckDictionary","Proto class found with the short name: %s",shortname);
      ++failedtest;
   }
   pClass = TClassTable::GetProto(longname);
   if (!pClass) {
      Error("CheckDictionary","Proto class not found with the long name (and normalization): %s",longname);
      ++failedtest;
   }
   return failedtest;
}

int CheckDictionary() {
   int failedtest = 0;

   TFile *pcm = new TFile("namingMatches_cxx_ACLiC_dict_rdict.pcm");
   if (!pcm) {
      Error("CheckDictionary","Can not open root pcm file: %s","namingMatches_cxx_ACLiC_dict_rdict.pcm");
      return 1;
   }
   TObjArray *proto; pcm->GetObject("__ProtoClasses",proto);
   if (!proto) {
      Error("CheckDictionary","Can not load list of proto classes.");
      return 1;
   }
   failedtest += CheckDictionary(proto,"vector<Wrapper<int,Object> >","vector<Wrapper<int>");
//   failedtest += CheckDictionary(proto,"atomic<vector<Wrapper<int,Object> >*>","atomic<vector<Wrapper<int> >*>");
   failedtest += CheckDictionary(proto,"list<Wrapper<int,Object> >","list<Wrapper<int> >");

   return failedtest;
}

int CheckGetClass(const char *longname, const char *shortname) {
   int failedtest = 0;

   TClass *c = TClass::GetClass(shortname);
   if (!c) {
      Error("CheckGetClass","Can not find %s TClass via its short name: %s",longname,shortname);
      ++failedtest;
   } else if (strcmp(c->GetName(),longname)) {
      Error("CheckGetClass","Class found via its short name: %s but the name is not correct (%s instead of %s)",
            shortname,c->GetName(),longname);
      ++failedtest;
   }
   c = TClass::GetClass(longname);
   if (!c) {
      Error("CheckGetClass","Can not find %s TClass via its long name",longname);
      ++failedtest;
   } else if (strcmp(c->GetName(),longname)) {
      Error("CheckGetClass","Class found via its long name: %s but the name is not correct %s",
            longname,c->GetName());
      ++failedtest;
   }
   return failedtest;
}

int CheckGetClass() {
   int failedtest = 0;

   failedtest += CheckGetClass("vector<Wrapper<int,Object> >","vector<Wrapper<int> >");
   failedtest += CheckGetClass("atomic<vector<Wrapper<int,Object> >*>","atomic<vector<Wrapper<int> >*>");
   failedtest += CheckGetClass("list<Wrapper<int,Object> >","list<Wrapper<int> >");

   return failedtest;
}

int execNamingMatches() {
   int failedtest = 0;

   gROOT->ProcessLine(".L namingMatches.cxx+s");

   THashTable result;
   TClass *c = TClass::GetClass("Holder");
   if (!c) {
      Error("execNamingMatches","Can not find Holder TClass");
      ++failedtest;
   } else {
      c->GetMissingDictionaries(result,true);
      if (result.GetEntries()) {
         Error("execNamingMatches","Found missing dictionaries:");
         result.ls();
         ++failedtest;
      }
   }
   c = TClass::GetClass("HolderAuto");
   if (!c) {
      Error("execNamingMatches","Can not find HolderAuto TClass");
      ++failedtest;
   } else {
      c->GetMissingDictionaries(result,true);
      if (result.GetEntries()) {
         Error("execNamingMatches","Found missing dictionaries:");
         result.ls();
         ++failedtest;
      }
   }

   failedtest += CheckDictionary();

   failedtest += CheckGetClass();

   return failedtest;
}
