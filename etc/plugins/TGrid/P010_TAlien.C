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
               tmp.ReplaceAll("$LD_LIBRARY_PATH","");
               gSystem->Unsetenv(strVar->GetString().Data());
               gSystem->Setenv(strVar->GetString().Data(),tmp.Data());
               if (gDebug>0) {printf("P010_TAlien.C: setting environemnt %s=\"%s\"\n", strVar->GetString().Data(),tmp.Data());}
               if (!strVar->GetString().CompareTo("GCLIENT_SERVER_LIST")) {
                  gSystem->Unsetenv("alien_API_SERVER_LIST");
                  gSystem->Setenv("alien_API_SERVER_LIST",tmp.Data());
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
                     if (gDebug>0) {printf("P010_TAlien.C: setting environemnt %s=\"%s\"\n", key.Data(),val.Data());}
                     gSystem->Unsetenv(key.Data());
                     gSystem->Setenv(key.Data(),val.Data());
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
               if (gDebug>0) {printf("P010_TAlien.C: setting environemnt %s=\"%s\"\n", key.Data(),val.Data());}
               gSystem->Unsetenv(key.Data());
               gSystem->Setenv(key.Data(),val.Data());
            }
         }
      }
   }
}

void P010_TAlien()
{
   // you can enforce 
   if ((!gSystem->Getenv("GBBOX_ENVFILE")) || 
       ( gSystem->Getenv("ALIEN_SOURCE_GCLIENT_ENV")) ||
       (!gSystem->Getenv("ALIEN_SKIP_GCLIENT_ENV")) ) {
      SetAliEnSettings();
   }
#ifdef __APPLE__
   const char* hlib = "libRAliEn.so";
   if (gSystem->Load(hlib)>=0) {
#else
   const char* hlib = "libgapiUI.so";
   if (gSystem->Load(hlib)>=0) {
#endif
      gPluginMgr->AddHandler("TGrid", "^alien", "TAlien",
                             "RAliEn", "TAlien(const char*,const char*,const char*,const char*)");
   } else {
      Error("P010_TAlien","Please fix your library search path to be able to load %s!",hlib);
   }
}


