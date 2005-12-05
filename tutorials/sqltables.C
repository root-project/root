
// example configuration for MySQL 4.1
const char* dbname = "mysql://host.domain/test";
const char* username = "user";
const char* userpass = "pass";

// example configuration for Oracle 9i
//const char* dbname = "oracle://host.domain/db-test";
//const char* username = "user";
//const char* userpass = "pass";


void sqltables() 
{
    tables_write();
    tables_read();
}


void tables_write() 
{
   // first connect to data base
   // recreate deletes all your tables !!!! 
   TSQLFile* f = new TSQLFile(dbname, "recreate", username, userpass);
   if (f->IsZombie()) { delete f; return; }

   // lets first write histogram
   TH1I* h1 = new TH1I("histo1","histo title", 1000, -4., 4.);
   h1->FillRandom("gaus",10000);
   h1->Write("histo");
   h1->SetDirectory(0);

   // here we create list of objects and store them as single key
   // without kSingleKey all TBox objects will appear as separate keys
   TList* arr = new TList;
   for(Int_t n=0;n<10;n++) {
      TBox* b = new TBox(n*10,n*100,n*20,n*200);  
      arr->Add(b, Form("option_%d_option",n));
   }
   arr->Write("arr",TObject::kSingleKey);
 
   // dummy, but possible situation
   // just TObject instance 
   TObject* t = new TObject;
   t->Write("tobj");

   // when file is closed (deleted), StreamerInfos will be stored to database
   delete f;
}


void tables_read() 
{
   // now open connection to database for read-only 
   TSQLFile* f = new TSQLFile(dbname, "open", username, userpass);
   if (f->IsZombie()) { delete f; return; }
   
   // see list of files
   f->ls();
   
   // get histogram from DB and draw it
   TH1* h1 = (TH1*) f->Get("histo");
   if (h1!=0) {
       h1->SetDirectory(0);
       h1->Draw();
   }
   
   // get TList with other objects
   TObject* obj = f->Get("arr");
   if (obj!=0) obj->Print("*");
   delete obj;

   // and just read TObject 
   obj = f->Get("tobj");
   if (obj!=0) obj->Print("*");
   delete obj;
   
   // close connection to database
   delete f;
}

