/// \file
/// \ingroup tutorial_sql
/// This is an example illustrating how the TSQLFile class can be used.
/// Histogram, list of TBox and clones array of TBox objects are stored
/// to TSQLFile and read back.
/// Except for the specific TSQLFile configuration, the TSQLFile functionality
/// is absolutely similar to a normal root TFile
///
/// \macro_code
///
/// \author Sergey Linev

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
   // "recreate" option delete all your tables !!!!
   TSQLFile* f = new TSQLFile(dbname, "recreate", username, userpass);
   if (f->IsZombie()) { delete f; return; }

   // you can change configuration only until first object
   // is written to TSQLFile
   f->SetUseSuffixes(kFALSE);
   f->SetArrayLimit(1000);
   f->SetUseIndexes(1);
//   f->SetTablesType("ISAM");
//   f->SetUseTransactions(kFALSE);

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
   arr->Write("list",TObject::kSingleKey);

   // clones array is also stored as single key
   TClonesArray clones("TBox",10);
   for(int n=0;n<10;n++)
       new (clones[n]) TBox(n*10,n*100,n*20,n*200);
   clones.Write("clones",TObject::kSingleKey);

   // close connection to database
   delete f;
}


void tables_read()
{
   // now open connection to database for read-only
   TSQLFile* f = new TSQLFile(dbname, "open", username, userpass);
   if (f->IsZombie()) { delete f; return; }

   // see list of keys
   f->ls();

   // get histogram from DB and draw it
   TH1* h1 = (TH1*) f->Get("histo");
   if (h1!=0) {
       h1->SetDirectory(0);
       h1->Draw();
   }

   // get TList with other objects
   TObject* obj = f->Get("list");
   cout << "Printout of TList object" << endl;
   if (obj!=0) obj->Print("*");
   delete obj;

   // and get TClonesArray
   obj = f->Get("clones");
   cout << "Printout of TClonesArray object" << endl;
   if (obj!=0) obj->Print("*");
   delete obj;

   // this is query to select data of hole class from different tables
   cout << "================ TBox QUERY ================ " << endl;
   cout << f->MakeSelectQuery(TBox::Class()) << endl;
   cout << "================ END of TBox QUERY ================ " << endl;

   cout << "================== TH1I QUERY ================ " << endl;
   cout << f->MakeSelectQuery(TH1I::Class()) << endl;
   cout << "================ END of TH1I QUERY ================ " << endl;

   // close connection to database
   delete f;
}
