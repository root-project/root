// @(#)root/tree:$Id$
// Author: Philippe Canal and al. 08/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeSQL                                                             //
//                                                                      //
// Implement TTree for a SQL backend                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <Riostream.h>
#include <vector>
#include <map>
#include <stdlib.h>

#include "TString.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TError.h"
#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TBranch.h"

#include "TSQLRow.h"
#include "TSQLResult.h"
#include "TSQLServer.h"

#include "TTreeSQL.h"
#include "TBasketSQL.h"

ClassImp(TTreeSQL)

//______________________________________________________________________________
TTreeSQL::TTreeSQL(TSQLServer *server, TString DB, const TString& table) :
   TTree(table.Data(), "Database read from table: " + table, 0), fDB(DB),
   fTable(table.Data()),
   fResult(0), fRow(0),
   fServer(server),
   fBranchChecked(kFALSE)
{
   // Constructor with an explicit TSQLServer

   fCurrentEntry = -1;
   fQuery = TString("Select * from " + fTable);
   fEntries = 0;

   if (fServer==0) {
      Error("TTreeSQL","No TSQLServer specified");
      return;
   }
   if (CheckTable(fTable.Data())) {
      Init();
   }
}

//______________________________________________________________________________
TBranch* TTreeSQL::BranchImp(const char *, const char *,
                             TClass *, void *, Int_t ,
                             Int_t )
{
   // Not implemented yet

   Fatal("BranchImp","Not implemented yet");
   return 0;
}

//______________________________________________________________________________
TBranch* TTreeSQL::BranchImp(const char *, TClass *,
                             void *, Int_t , Int_t )
{
   // Not implemented yet

   Fatal("BranchImp","Not implemented yet");
   return 0;
}

//______________________________________________________________________________
Int_t TTreeSQL::Branch(TCollection *, Int_t,
                       Int_t, const char *)
{
   // Not implemented yet

   Fatal("Branch","Not implemented yet");
   return 0;
}

//______________________________________________________________________________
Int_t TTreeSQL::Branch(TList *, Int_t, Int_t)
{
   // Not implemented yet

   Fatal("Branch","Not implemented yet");
   return 0;
}

//______________________________________________________________________________
Int_t TTreeSQL::Branch(const char *, Int_t ,
                       Int_t)
{
   // Not implemented yet

   Fatal("Branch","Not implemented yet");
   return 0;
}

//______________________________________________________________________________
TBranch* TTreeSQL::Bronch(const char *, const char *, void *,
                          Int_t, Int_t)
{
   // Not implemented yet

   Fatal("Bronc","Not implemented yet");
   return 0;
}

//______________________________________________________________________________
TBranch* TTreeSQL::BranchOld(const char *, const char *,
                             void *, Int_t, Int_t)
{
   // Not implemented yet

   Fatal("BranchOld","Not implemented yet");
   return 0;
}

//______________________________________________________________________________
TBranch *TTreeSQL::Branch(const char *, const char *, void *,
                          Int_t, Int_t)
{
   // Not implemented yet

   Fatal("Branch","Not implemented yet");
   return 0;
}

//______________________________________________________________________________
TBranch * TTreeSQL::Branch(const char *name, void *address,
                           const char *leaflist, Int_t bufsize)
{
   // Create a branch

   Int_t nb = fBranches.GetEntriesFast();
   TBranch *branch;
   TString brName;

   for (int i=0;i<nb;i++) {
      branch = (TBranch*)fBranches.UncheckedAt(i);
      brName = branch->GetName();
      if (brName.Index(name) == 0) {
         // Now if the branch exists in db, root gives a warning and exit
         // Dealing with duplicate branch has been done, but not tested yet.
         // So if you want to allow duplicate branch, just comment Fatal() line and uncomment commented
         // below Fatal() line

         Fatal("Branch()", "Duplicate branch!!!");

         /* Commented. If uncommented, should comment Fatal line.
         // this is a duplicate branch. So reset data structure memory address and return.
         branch->SetAddress(address);
         return branch;
         */
      }
   }
   return TTree::Branch(name, address, leaflist, bufsize);
}

//______________________________________________________________________________
void TTreeSQL::CheckBasket(TBranch *branch)
{
   // Check if the basket is properly setup

   TBasketSQL* basket = (TBasketSQL *)branch->GetBasket(0);

   if (basket==0) {
      basket = (TBasketSQL*)CreateBasket(branch);
      if (basket==0) return;
      //++(branch->fNBaskets);
      branch->GetListOfBaskets()->AddAtAndExpand(basket,0);
   }
   TBuffer * buffer = basket->GetBufferRef();

   if(buffer == 0){
      vector<Int_t> *columns = GetColumnIndice(branch);
      if (columns) basket->CreateBuffer(branch->GetName(),"A", columns, branch, &fResult);
   }

   Int_t nb = branch->GetListOfBranches()->GetEntriesFast();
   for (int i=0;i<nb;i++) {
      TBranch * subbranch = (TBranch*)branch->GetListOfBranches()->UncheckedAt(i);
      if(subbranch) CheckBasket(subbranch);
   }
}

//______________________________________________________________________________
Bool_t TTreeSQL::CheckBranch(TBranch * tb)
{
   // Check if the table has a column corresponding the branch
   // and that the resultset are properly setup

   if (fServer==0) {
      return kFALSE;
   }
   TString leafName;
   TLeaf *leaf;
   Int_t nl;
   TString str = "";
   TString typeName = "";

   if (!tb) return kFALSE;

   TBasketSQL *basket = (TBasketSQL *)tb->GetBasket(0);
   if (!basket) return kFALSE;

   TSQLResult *rs = basket->GetResultSet();
   if (!rs) {
      Error("CheckBranch","%s has basket but no resultset yet",tb->GetName());
      return kFALSE;
   }

   nl = tb->GetNleaves();

   for(int j=0;j<nl;j++) {
      leaf = (TLeaf*)tb->GetListOfLeaves()->UncheckedAt(j);
      typeName = leaf->GetTypeName();
      typeName = ConvertTypeName(leaf->GetTypeName());
      leafName = leaf->GetName();
      str = "";
      str = tb->GetName();
      str += "__";
      str += leafName;

      for (int i=0; i< rs->GetFieldCount(); ++i) {
         if (str.CompareTo(rs->GetFieldName(i),TString::kIgnoreCase) == 0) return kTRUE;
      }
      // We assume that if ONE of the leaf is in the table, then ALL the leaf are in
      // the table.
      // TODO: this assumption is harmful if user changes branch structure while keep its name
      CreateBranch(str, typeName);
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TTreeSQL::CheckTable(const TString &table) const
{
   // Check the table exist in the database

   if (fServer==0) return kFALSE;
   TSQLResult * tables = fServer->GetTables(fDB.Data(),table);
   TSQLRow * row = 0;
   while( (row = tables->Next()) ) {
      if(table.CompareTo(row->GetField(0),TString::kIgnoreCase)==0){
         return kTRUE;
      }
   }
   // The table is a not a permanent table, let's see if it is a 'temporary' table
   Int_t before = gErrorIgnoreLevel;
   gErrorIgnoreLevel = kFatal;
   TSQLResult *res = fServer->GetColumns(fDB.Data(),table);
   if (res) {
      delete res;
      return kTRUE;
   }
   gErrorIgnoreLevel = before;

   return kFALSE;
}

//______________________________________________________________________________
TString TTreeSQL::ConvertTypeName(const TString& typeName )
{
   // Convert from ROOT typename to SQL typename

   TString tn = "";

   if(typeName == "Char_t"){
      tn = "TEXT";
   }
   else if(typeName == "Int_t") {
      tn = "INTEGER";
   }
   else if(typeName == "Short_t") {
      tn = "SMALLINT";
   }
   else if( typeName == "UShort_t") {
      tn = "SMALLINT UNSIGNED";
   }
   else if(typeName == "Float_t"){
      tn = "FLOAT";
   }
   else if(typeName == "Float16_t"){
      tn = "FLOAT";
   }
   else if(typeName == "Double_t"){
      tn = "DOUBLE";
   }
   else if(typeName == "Double32_t"){
      tn = "FLOAT";
   }
   else if(typeName == "UInt_t") {
      tn = "INT UNSIGNED";
   }
   else if( typeName == "Long_t") {
      tn = "INTEGER";
   }
   else if( typeName == "ULong_t") {
      tn = "INTEGER UNSIGNED";
   }
   else if( typeName == "Long64_t") {
      tn = "BIGINT";
   }
   else if( typeName == "ULong64_t") {
      tn = "BIGINT UNSIGNED";
   }
   else if( typeName == "Bool_t") {
      tn = "BOOL";
   }
   else {
      Error("ConvertTypeName","TypeName (%s) not found",typeName.Data());
      return "";
   }

   return tn;
}

//______________________________________________________________________________
TBasket * TTreeSQL::CreateBasket(TBranch * tb)
{
   // Create a TBasketSQL

   if (fServer==0) {
      Error("CreateBasket","No TSQLServer specified");
      return 0;
   }
   vector<Int_t> *columnVec = GetColumnIndice(tb);
   if (columnVec) {
      return new TBasketSQL(tb->GetName(), tb->GetName(), tb,
                            &fResult, &fInsertQuery, columnVec, &fRow);
   } else {
      return 0;
   }
}

//______________________________________________________________________________
void TTreeSQL::CreateBranch(const TString &branchName, const TString &typeName)
{
   // Create the column(s) in the database that correspond to the branch/

   if (fServer==0) {
      Error("CreateBranch","No TSQLServer specified");
      return;
   }
   TString alterSQL = "";
   alterSQL = "";
   alterSQL = "ALTER TABLE ";
   alterSQL += fTable.Data();
   alterSQL += " ADD ";
   alterSQL += branchName.Data();;
   alterSQL += " ";
   alterSQL += typeName;
   alterSQL += " ";

   fServer->Query(alterSQL);
}

//_________________________________________________________________________
TString TTreeSQL::CreateBranches(TSQLResult * rs)
{
   // determine leaf description string

   if(!rs) return "";

   Int_t rows;
   TString type;
   TString res;
   TString branchName;
   TString leafName;
   Int_t prec=0;
   TBranch * br = 0;
   rows = rs->GetRowCount();
   TString decl;
   TString prevBranch;

   for( int i=0; i < rows; ++i ) {
      TSQLRow * row = rs->Next();
      type = row->GetField(1);
      Int_t index = type.First('(');
      if(index>0){
         prec = atoi(type(index+1,type.First(')')-1).Data());
         type = type(0,index);
      }
      branchName = row->GetField(0);
      Int_t pos;
      if ((pos=branchName.Index("__"))!=kNPOS) {
         leafName = branchName(pos+2,branchName.Length());
         branchName.Remove(pos);
      } else {
         leafName = branchName;
      }
      if (prevBranch.Length()) {
         if (prevBranch != branchName) {
            // new branch let's flush.
            if (decl.Length()) decl.Remove(decl.Length()-1);
            br = TTree::Branch(prevBranch,0,decl);
            br->ResetAddress();

            (br->GetBasketEntry())[0] = 0;
            (br->GetBasketEntry())[1] = fEntries;

            br->SetEntries(fEntries);

            //++(br->fNBaskets);
            br->GetListOfBaskets()->AddAtAndExpand(CreateBasket(br),0);

            prevBranch = branchName;
            decl = "";
         }
      } else {
         prevBranch = branchName;
      }

      if(type.CompareTo("varchar",TString::kIgnoreCase)==0 || type.CompareTo("varchar2",TString::kIgnoreCase)==0 || type.CompareTo("char",TString::kIgnoreCase)==0 ) { 
         char siz[6];
         snprintf(siz,6,"[%d]",prec);
         decl.Append( leafName+siz+"/C:" );
      }
      else if(type.CompareTo("int",TString::kIgnoreCase)==0){
         decl.Append( leafName+"/I:" );
      }
      else if( type.CompareTo("date",TString::kIgnoreCase)==0 ||
               type.CompareTo("time",TString::kIgnoreCase)==0 ||
               type.CompareTo("timestamp",TString::kIgnoreCase)==0 ) {
         decl.Append( leafName+"/I:" );
      }
      else if(type.CompareTo("bit",TString::kIgnoreCase)==0 ||
              type.CompareTo("tinyint",TString::kIgnoreCase)==0 ||
              type.CompareTo("smallint",TString::kIgnoreCase)==0 ) {
         decl.Append( leafName+"/i:" );
      }
      else if(type.CompareTo("real",TString::kIgnoreCase)==0 || type.CompareTo("longvarchar",TString::kIgnoreCase)==0 || type.CompareTo("longvarbinary",TString::kIgnoreCase)==0 || type.CompareTo("varbinary",TString::kIgnoreCase)==0 ){
         decl.Append( leafName+"/S:" );
      }

      //   case kLONGVARCHAR: // not resolved yet how to handle
      // case kLONGVARBINARY:
      //case kVARBINARY:
      //  break;
      else /*if(type.CompareTo("bigint",TString::kIgnoreCase)==0 || type.CompareTo("decimal",TString::kIgnoreCase)==0 || type.CompareTo("numeric",TString::kIgnoreCase)==0 || type.CompareTo("double",TString::kIgnoreCase)==0 ||
      type.CompareTo("float",TString::kIgnoreCase)==0 )*/{

         decl.Append( leafName+"/F:" );
      }

   }

   // new branch let's flush.
   if (decl.Length()) decl.Remove(decl.Length()-1);
   if (prevBranch.Length()) {
      br = TTree::Branch(prevBranch,0,decl);
      br->ResetAddress();

      (br->GetBasketEntry())[0] = 0;
      (br->GetBasketEntry())[1] = fEntries;
      br->SetEntries(fEntries);
      br->GetListOfBaskets()->AddAtAndExpand(CreateBasket(br),0);
   }

   if(!res.IsNull()) res.Resize(res.Length()-1);   // cut off last ":"
   return res;
}

//______________________________________________________________________________
Bool_t TTreeSQL::CreateTable(const TString &table)
{
   // Create the database table corresponding to this TTree.

   if (fServer==0) {
      Error("CreateTable","No TSQLServer specified");
      return false;
   }
   Int_t i, j;
   TString branchName, leafName, typeName;
   TString createSQL, alterSQL, str;
   Int_t nb = fBranches.GetEntriesFast();
   Int_t nl = 0;

   TBranch *branch;
   TLeaf *leaf;

   for (i=0;i<nb;i++) {
      branch = (TBranch*)fBranches.UncheckedAt(i);
      branchName = branch->GetName();
      nl = branch->GetNleaves();
      for(j=0;j<nl;j++) {
         leaf = (TLeaf*)branch->GetListOfLeaves()->UncheckedAt(j);
         leafName = leaf->GetName();
         typeName = ConvertTypeName(leaf->GetTypeName());
         // length = leaf->GetLenStatic();

         if(i == 0 && j == 0) {
            createSQL = "";
            createSQL += "CREATE TABLE ";
            createSQL += table;
            createSQL += " (";
            createSQL += branchName;
            createSQL += "__";
            createSQL += leafName;
            createSQL += " ";
            createSQL += typeName;
            createSQL += " ";
            createSQL += ")";

            TSQLResult *sres = fServer->Query(createSQL.Data());
            if (!sres) {
               Error("CreateTable","May have failed");
               return false;
            }
         }
         else {
            str = "";
            str = branchName;
            str += "__";
            str += leafName;
            CreateBranch(str, typeName);
         } //else
      }  // inner for loop
   } // outer for loop
   // retrieve table to initialize fResult
   delete fResult;
   fResult = fServer->Query(fQuery.Data());
   return (fResult!=0);
}

//______________________________________________________________________________
void TTreeSQL::Init()
{
   // Initializeation routine

   fCurrentEntry = -1;

   GetEntries();

   delete fResult;
   fResult = fServer->Query(fQuery.Data());
   if(!fResult) return;

   CreateBranches(fServer->GetColumns(fDB,fTable));
}

//______________________________________________________________________________
Int_t TTreeSQL::Fill()
{
   // Copy the information from the user object to the TTree

   Int_t nb = fBranches.GetEntriesFast();
   TString typeName;
   TBranch *branch;

   if (fServer==0) return 0;

   if(!CheckTable(fTable.Data())) {
      if (!CreateTable(fTable.Data())) {
         return -1;
      }
   }

   PrepEntry(fEntries);

   for (int i=0;i<nb;i++) {
      branch = (TBranch*)fBranches.UncheckedAt(i);
      CheckBasket(branch);
   }

   if (!fBranchChecked) {
      for(int i=0;i<nb;i++) {
         branch = (TBranch*)fBranches.UncheckedAt(i);
         if (!CheckBranch(branch)) {
            Error("Fill","CheckBranch for %s failed",branch->GetName());
         }
      }
      fBranchChecked = kTRUE;
   }
   ResetQuery();

   TTree::Fill();

   if (fInsertQuery[fInsertQuery.Length()-1]!='(') {
      fInsertQuery.Remove(fInsertQuery.Length()-1);
      fInsertQuery += ")";
      TSQLResult *res = fServer?fServer->Query(fInsertQuery):0;

      if (res) {
         return res->GetRowCount();
      }
   }
   return -1;
}

//______________________________________________________________________________
vector<Int_t> *TTreeSQL::GetColumnIndice(TBranch *branch)
{
   // Return a vector of columns index corresponding to the
   // current SQL table and the branch given as argument
   // Returns 0 if no columns indices is found
   // Otherwise returns a pointer to a vector to be deleted by the caller

   if (!CheckTable(fTable)) return 0;

   vector<Int_t> *columns = new vector<Int_t>;

   Int_t nl = branch->GetNleaves();

   vector<TString> names;

   TSQLResult *rs = fServer->GetColumns(fDB,fTable);
   if (rs==0) { delete columns; return 0; }
   Int_t rows = rs->GetRowCount();

   pair<TString,Int_t> value;

   for (Int_t i=0;i<rows;++i) {
      TSQLRow *row = rs->Next();
      names.push_back( row->GetField(0) );
      delete row;
   }
   delete rs;

   for(int j=0;j<nl;j++) {

      Int_t col = -1;
      TLeaf *leaf = (TLeaf*)branch->GetListOfLeaves()->UncheckedAt(j);
      TString leafName = leaf->GetName();
      TString str;

      str = "";
      str = branch->GetName();
      str += "__";
      str += leafName;
      for (Int_t i=0;i<rows;++i) {
         if (str.CompareTo(names[i],TString::kIgnoreCase)==0) {
            col = i;
            break;
         }
      }
      if (col<0) {
         str = leafName;
         for (Int_t i=0;i<rows;++i) {
            if (str.CompareTo(names[i],TString::kIgnoreCase)==0) {
               col = i;
               break;
            }
         }         
      }
      if(col>=0){
         columns->push_back(col);
      } else Error("GetColumnIndice","Error finding column %d %s",j,str.Data());
   }
   if (columns->empty()) {
      delete columns; return 0;
   } else
      return columns;
}

//______________________________________________________________________________
Long64_t  TTreeSQL::GetEntries() const
{
   // Get the number of rows in the database

   if (fServer==0) return GetEntriesFast();
   if (!CheckTable(fTable.Data())) return 0;

   TTreeSQL* thisvar = (TTreeSQL*)this;

   // What if the user already started to call GetEntry
   // What about the initial value of fEntries is it really 0?

   TString counting = "select count(*) from " + fTable;
   TSQLResult *count = fServer->Query(counting);

   if (count==0) {
      thisvar->fEntries = 0;
   } else {
      TString val = count->Next()->GetField(0);
      Long_t ret;
      sscanf(val.Data(), "%ld",&(ret) );
      thisvar->fEntries = ret;
   }
   return fEntries;
}

//______________________________________________________________________________
Long64_t  TTreeSQL::GetEntriesFast()    const
{
   // Return the number of entries as of the last check.
   // Use GetEntries for a more accurate count.

   return fEntries;
}

//______________________________________________________________________________
Int_t TTreeSQL::GetEntry(Long64_t entry, Int_t getall)
{
   // Load the data for the entry from the database.

   if (PrepEntry(entry)>=0) return TTree::GetEntry(entry,getall);
   else return -1;
}

//______________________________________________________________________________
Long64_t TTreeSQL::LoadTree(Long64_t entry)
{
   // Setup the tree to the load the specified entry.

   fReadEntry = entry;
   return PrepEntry(entry);
}

//______________________________________________________________________________
Long64_t TTreeSQL::PrepEntry(Long64_t entry)
{
   // Make sure the server and result set are setup for the requested entry

   if (entry < 0 || entry >= fEntries || fServer==0) return 0;
   fReadEntry = entry;

   if(entry == fCurrentEntry) return entry;

   if(entry < fCurrentEntry || fResult==0){
      delete fResult;
      fResult = fServer->Query(fQuery.Data());
      fCurrentEntry = -1;
   }

   Bool_t reset = false;
   while ( fCurrentEntry < entry ) {
      ++fCurrentEntry;
      delete fRow;
      fRow = fResult->Next();
      if (fRow==0 && !reset) {
         delete fResult;
         fResult = fServer->Query(fQuery.Data());
         fCurrentEntry = -1;
         reset = true;
      }
   }
   if (fRow==0) return -1;
   return entry;
}

//______________________________________________________________________________
// void TTreeSQL::LoadNumberEntries()
// {
//    R__ASSERT(0);

//    fResult =    fServer->Query(fQuery.Data());
//    fEntries=0;

//    while(fResult->Next()){
//       fEntries++;
//    }
//    fResult =    fServer->Query(fQuery.Data());
// }

//______________________________________________________________________________
void TTreeSQL::Refresh()
{
   //  Refresh contents of this Tree and his branches from the current
   //  Tree status in the database
   //  One can call this function in case the Tree on its file is being
   //  updated by another process

   // Note : something to be done?
   GetEntries(); // Re-load the number of entries
   fCurrentEntry = -1;
   delete fResult; fResult = 0;
   delete fRow; fRow = 0;
}

//______________________________________________________________________________
void TTreeSQL::ResetQuery()
{
   // Reset the internal query

   fInsertQuery = "INSERT INTO " + fTable + " VALUES (";
}


