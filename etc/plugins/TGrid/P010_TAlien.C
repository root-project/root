#include <iostream>

void SetAliEnSettings()
{
   // Routine to load settings from an AliEn environment file.

   ifstream fileIn;
   fileIn.open(Form("/tmp/gclient_env_%d",gSystem->GetUid()));
   if (gDebug>0) {printf("P010_TAlien.C: parsing /tmp/gclient_env_$UID\n");}
   TString lineS,tmp;
   char line[4096];

   while (fileIn.good()){
      fileIn.getline(line,4096,'\n');
      lineS = line;
      if (lineS.IsNull()) continue;
      if (lineS.Contains("export ")) {
         lineS.ReplaceAll("export ","");

         TObjArray* array = lineS.Tokenize("=");

         if (array->GetEntries() == 2) {
            TObjString *strVar = (TObjString *) array->At(0);
            TObjString *strVal = (TObjString *) array->At(1);

            if ((strVar)&&(strVal)) {
               tmp = strVal->GetString();
               tmp.ReplaceAll("\"","");
               tmp.ReplaceAll("$LD_LIBRARY_PATH",gSystem->Getenv("LD_LIBRARY_PATH"));
               tmp.ReplaceAll("$DYLD_LIBRARY_PATH",gSystem->Getenv("DYLD_LIBRARY_PATH"));
               tmp.ReplaceAll(" ","");
               gSystem->Unsetenv(strVar->GetString());
               gSystem->Setenv(strVar->GetString(), tmp);
               if (gDebug>0) {
                  Info("P010_TAlien", "setting environment %s=\"%s\"", strVar->GetString().Data(), tmp.Data());
               }
               if (!strVar->GetString().CompareTo("GCLIENT_SERVER_LIST")) {
                  gSystem->Unsetenv("alien_API_SERVER_LIST");
                  gSystem->Setenv("alien_API_SERVER_LIST", tmp);
               }
            }
            if (array) {
               delete array;
               array = 0 ;
            }
         } else {
            // parse the MONA_ stuff
            TObjArray* array = lineS.Tokenize("\" ");
            TString key="";
            TString val="";
            for (int i=0; i< array->GetEntries(); i++) {
               if ( ((TObjString*) array->At(i))->GetString().Contains("=")) {
                  if (key.Length() && val.Length()) {
                     val.Resize(val.Length()-1);
                     if (gDebug>0) {
                        Info("P010_TAlien", "setting environment %s=\"%s\"", key.Data(), val.Data());
                     }
                     gSystem->Unsetenv(key);
                     gSystem->Setenv(key, val);
                     key="";
                     val="";
                  }
                  key = ((TObjString*) array->At(i))->GetString();
                  key.ReplaceAll("=","");
               } else {
                  val+=((TObjString*) array->At(i))->GetString();
                  val+=" ";
               }
            }
            if (key.Length() && val.Length()) {
               if (gDebug>0) {
                  Info("P010_TAlien", "setting environment %s=\"%s\"", key.Data(), val.Data());
               }
               gSystem->Unsetenv(key);
               gSystem->Setenv(key, val);
            }
         }
      }
   }
}

void P010_TAlien()
{
   TString configfeatures = gROOT->GetConfigFeatures();
   TString ralienpath = gSystem->Getenv("ROOTSYS");
   ralienpath += "/lib/"; ralienpath += "libRAliEn.so";

   // only if ROOT was compiled with enable-alien we do library setup and configure a handler
   if ((!gSystem->AccessPathName(ralienpath))) || (configfeatures.contains("alien"))) {
      // you can enforce
      if ((!gSystem->Getenv("GBBOX_ENVFILE")) ||
          ( gSystem->Getenv("ALIEN_SOURCE_GCLIENT_ENV")) ||
          (!gSystem->Getenv("ALIEN_SKIP_GCLIENT_ENV")) ) {
       SetAliEnSettings();
      }

      if ( ((gSystem->Load("libgapiUI.so")>=0) && (gSystem->Load("libRAliEn.so")>=0)))  {
         gPluginMgr->AddHandler("TGrid", "^alien", "TAlien",
                                "RAliEn", "TAlien(const char*,const char*,const char*,const char*)");
      } else {
         Error("P010_TAlien","Please fix your loader path environment variable to be able to load libRAliEn.so");
      }
  }
}
