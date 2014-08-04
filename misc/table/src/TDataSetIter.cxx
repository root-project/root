// @(#)root/table:$Id$
// Author: Valery Fine(fine@mail.cern.ch)   03/07/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riosfwd.h"
#include "Riostream.h"

#include "TDataSetIter.h"
#include "TBrowser.h"
#include "TSystem.h"

#ifndef WIN32
# ifndef HASSTRCASE
#  define HASSTRCASE
# endif
#endif

#ifndef HASSTRCASE
#  define strcasecmp(arg1,arg2) stricmp(arg1,arg2)
#endif

TDataSet *TDataSetIter::fgNullDataSet = (TDataSet *)(-1);

ClassImp(TDataSetIter)

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDataSetIter                                                         //
//                                                                      //
// TDataSetIter is a class iterator to navigate TDataSet objects        //
// via 4 internal pointers :                                            //
//                                                                      //
//  1. fRootDataSet    - "root" dataset                                 //
//  2. fWorkingDataSet - Working dataset                                //
//  3. fDataSet        - the last selected TDataSet                     //
//  4. fNext           - TIter for the the list of the "root" dataset   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TDataSetIter::TDataSetIter(TDataSet *link, Bool_t dir)
{
   //to be documented
   fWorkingDataSet= fRootDataSet   =link;
   fMaxDepth      = fDepth         =1;
   fDataSet= fgNullDataSet ;
   fNext = link ? new TIter(link->GetCollection() ,dir):0;
   for(UInt_t i = 0; i < sizeof(fNextSet) / sizeof(TIter*); ++i) {
      fNextSet[i] = (TIter*)0;
   }
}

//______________________________________________________________________________
TDataSetIter::TDataSetIter(TDataSet *link, Int_t depth, Bool_t dir)
{
   //to be documented
   fRootDataSet = fWorkingDataSet = link;
   fMaxDepth    = depth;
   fDepth       = 1;
   fDataSet     = fgNullDataSet;
   fNext        = (link)? new TIter(link->GetCollection() ,dir):0;

   // Create a DataSet iterator to pass all nodes of the
   //     "depth"  levels
   //  of  TDataSet *link

   for(UInt_t i = 0; i < sizeof(fNextSet) / sizeof(TIter*); ++i) {
      fNextSet[i] = (TIter*)0;
   }
   if (fMaxDepth != 1) {
      fNextSet[0] = fNext;
      if (fMaxDepth > 100) fMaxDepth = 100;
      fDepth = 0;
   }
}

//______________________________________________________________________________
TDataSetIter::~TDataSetIter()
{
   //to be documented
   if (fMaxDepth != 1) {
      Int_t level = fDepth;
      if (level) level--;
      for (Int_t i = level;i>=0;i--) {
         TIter *s = fNextSet[i];
         if (s) delete s;
      }
   }
   else
      SafeDelete(fNext);
   fDepth = 0;
}


//______________________________________________________________________________
TDataSet *TDataSetIter::operator *() const
{
   //operator *
   return fDataSet == fgNullDataSet ? fWorkingDataSet : fDataSet;
}

//______________________________________________________________________________
TDataSet *TDataSetIter::GetNullSet()
{
   // return a fake pointer == -1 casted to (TDataSet *)
   return (TDataSet *)fgNullDataSet;
}

//______________________________________________________________________________
TDataSet *TDataSetIter::Add(TDataSet *set, TDataSet *dataset)
{
   ///////////////////////////////////////////////////////////////////////////////
   //                                                                           //
   // Add - adds the set to the dataset defined with the second parameters      //
   //                                                                           //
   // TDataSet dataset != 0 - Add the set to the TDataSet *dataset              //
   //                                                                           //
   //                     = 0 - (by default) to the current TDataSet defined    //
   //                          with fWorkingDataSet data member                 //
   //                                                                           //
   //  returns  the pointer to set is success or ZERO poiner                    //
   //  =======                                                                  //
   //                                                                           //
   //  Note: If this TDataSetIter is empty (i.e. Cwd() returns 0), the "set"    //
   //        becomes the "root" dataset of this iterator                        //                                                                         //
   ///////////////////////////////////////////////////////////////////////////////

   if (!set) return 0;
   TDataSet *s =  dataset;
   if (!s) s = Cwd();
   if (s) {
      s->Add(set);
      s = set;
   }
   else {
      //  make the coming dataset the current one for the iterator
      s = set;
      fRootDataSet    = s;
      fWorkingDataSet = s;
      if (fNext) {
         Error("Add","TDataSetIter.has been corrupted ;-!");
         delete fNext;
         fNext = 0;
      }
      fNext = new TIter(s->GetCollection() );
   }
   return s;
}

//______________________________________________________________________________
TDataSet *TDataSetIter::Add(TDataSet *dataset, const Char_t *path)
{
   ///////////////////////////////////////////////////////////////////////////////
   //                                                                           //
   // Add                                                                       //
   //                                                                           //
   // Char_t path != 0 - Add a TDataSet dataset to the TDataSet dataset         //
   //                    defined with "path"                                    //
   //              = 0 - (by default) to the current TDataSet defined           //
   //                     with fWorkingDataSet data member                      //
   //                                                                           //
   //  returns the dataset is success or ZERO pointer                           //
   //  =======                                                                  //
   //                                                                           //
   ///////////////////////////////////////////////////////////////////////////////
   if (!dataset) return 0;
   TDataSet *set = 0;
   if (path && strlen(path)) set = Find(path);
   return Add(dataset,set);
}

//______________________________________________________________________________
TDataSet *TDataSetIter::Cd(const Char_t *dirname){
   /////////////////////////////////////////////////////////////////////
   //                                                                 //
   // TDataSet *TDataSetIter::Cd(const Char_t *dirname)               //
   //                                                                 //
   // Change the current working directory to dirname                 //
   //                                                                 //
   // Returns the pointer to the new "working" TDataSet               //
   // =======   0,  if the new directory doesn't exist.               //
   //                                                                 //
   // Remark:  The name = ".." has a special meaning.                 //
   // ------   TDataSetIter::Cd("..") returns the parent set          //
   //          But one still can not use ".." as a legal part         //
   //          of the full path                                       //
   /////////////////////////////////////////////////////////////////////
   TDataSet *set = 0;
   if (strcmp(dirname,".."))
      set =  Find(dirname);
   else
      set = fWorkingDataSet->GetParent();
   if (set) fWorkingDataSet = set;
   return set;
}

//______________________________________________________________________________
TDataSet *TDataSetIter::Cd(TDataSet *ds)
{
   /////////////////////////////////////////////////////////////////////
   //                                                                 //
   // TDataSet *TDataSetIter::Cd(const TDataSet *ds)                  //
   //                                                                 //
   // Make:  Cwd() = ds;                                              //
   // Look for the first occurence of the "ds" pointer for the current//
   // TDataSet in respect of the Cwd() if any                         //
   //                                                                 //
   // Change the current working directory to ds if present           //
   //                                                                 //
   // Returns the pointer to the new "working" TDataSet (i.e. ds)     //
   // =======   0,  if the new directory doesn't exist.               //
   //                                                                 //
   /////////////////////////////////////////////////////////////////////
   TDataSet *nextSet = 0;
   if (Cwd()) {
      TDataSetIter next(Cwd(),0);
      while ( (nextSet = next()) )
         if (ds == nextSet) {fWorkingDataSet = ds; break;}
   }
   return nextSet;
}

//______________________________________________________________________________
TDataSet *TDataSetIter::Dir(Char_t *dirname)
{
   //
   // Print the names of the TDataSet objects for the datatset named with "dirname"
   // apart of TDataSet::Ls()  this method prints one level only
   //
   TDataSet *set = (TDataSet *)fWorkingDataSet;
   if (dirname) set = Find(dirname);
   if (set) set->ls();
   return set;
}

//______________________________________________________________________________
Int_t TDataSetIter::Du() const {
   // summarize dataset usage by Herb Ward proposal
   if (!fWorkingDataSet) return 0;
   TDataSetIter next(fWorkingDataSet,0);
   TDataSet *nextset = 0;
   Int_t count = 0;
   while((nextset = (count) ? next():fWorkingDataSet)) {
      count++;
      if (nextset->IsFolder()) std::cout << std::endl;
      TString path = nextset->Path();
      std::cout << std::setw(2) << next.GetDepth() << ". ";
      std::cout << path << std::setw(TMath::Max(Int_t(60-strlen(path.Data())),Int_t(0))) << "...";
      const Char_t *type = nextset->IsFolder() ? "directory" : "table" ;
      std::cout << std::setw(10) << type;
      std::cout  << " : " << std::setw(10) << nextset->GetTitle();
      std::cout << std::endl;
   }
   return count;
}

//______________________________________________________________________________
TDataSet  *TDataSetIter::FindByName(const Char_t *name,const Char_t *path,Option_t *opt)
{
   //to be documented
   return FindDataSet(name,path,opt);
}

//______________________________________________________________________________
TDataSet  *TDataSetIter::FindByTitle(const Char_t *title,const Char_t *path,Option_t *opt)
{
   //to be documented
   TString optt = "-t";
   optt += opt;
   return FindDataSet(title,path,optt.Data());
}

//______________________________________________________________________________
TDataSet *TDataSetIter::FindDataSet(const Char_t *name,const Char_t *path,Option_t *opt)
{
   //
   // FindDataSet looks for the object with the name supplied across dataset.
   //
   // name        - the "base" name title (with no path) of the TDataSet (see: opt = -t)
   // path        - path to start the search from (the current dataset "by default")
   // opt = "-i"  - case insensitive search
   //       "-t"  - first <name> parameter defines the object "title" rather the object "name"
   //
   // Note: If the name provided is not unique
   //       the first found is returned.
   //

   if (!name || !name[0]) return 0;
   if (strchr(name,'/')) {
      Error("FindDataSet","The name of the object <%s> can not contain any \"/\"",name);
      return 0;
   }

   Bool_t opti = opt ? strcasecmp(opt,"-i") == 0 : kFALSE;
   Bool_t optt = opt ? strcasecmp(opt,"-t") == 0 : kFALSE;

   TDataSet *startset = 0;
   if (path && strlen(path)) startset = Find(path);
   else                      startset = fWorkingDataSet;
   if (!startset) return 0;

   TDataSet *set = startset;
   if ( !((opti && strcasecmp( optt ? set->GetTitle() : set->GetName(),name) == 0 ) ||
      (strcmp(optt ? set->GetTitle() : set->GetName(),name) == 0)) )
   {
      TDataSetIter next(startset,0);
      while ((set = next()))
         if ( (opti && strcasecmp(optt ? set->GetTitle() : set->GetName(),name) == 0 ) ||
            (strcmp(optt ? set->GetTitle() : set->GetName(),name) == 0) )           break;
   }

   return set;
}

//______________________________________________________________________________
TDataSet *TDataSetIter::FindDataSet(TDataSet *set,const Char_t *path,Option_t *opt)
{
   //
   // Check whether the object does belong the TDataSet defined with "path"
   // opt = "-l"  - check the "reference" links only
   //       "-s"  - check the "structural" links only
   //             = "by default" - checks all links
   //
   if (!set) return 0;
   if (opt) {/* no used */}

   TDataSet *startset = 0;
   if (path) startset = Find(path);
   else      startset = fWorkingDataSet;
   if (!startset) return 0;

   TDataSetIter next(startset);
   TDataSet *nextSet = 0;
   while ( (nextSet = next()) )
      if (set == nextSet) break;

   return nextSet;
}
//______________________________________________________________________________
TObject *TDataSetIter::FindObject(const Char_t *name) const
{
   // This method is not recommended.
   // It is done to back TObject::FindObject method only.
   // One is recommnened to use FindByName method instead.
   return ((TDataSetIter *)this)->FindByName(name);
}

//______________________________________________________________________________
TObject  *TDataSetIter::FindObject(const TObject *dataset) const
{
   // This method is not recommended.
   // It is done to back TObject::FindObject method only.
   // One is recommended to use FindByName method instead.
   return ((TDataSetIter *)this)->FindByPointer((TDataSet *)dataset);
}
//______________________________________________________________________________
TDataSet *TDataSetIter::FindByPointer(TDataSet *set,const Char_t *path,Option_t *)
{
   //
   // Check whether the object does belong the TDataSet defined with "path"
   // opt = "-l"  - check the "reference" links only
   //       "-s"  - check the "structural" links only
   //             = "by default" - checks all links
   //
   if (!set) return 0;

   TDataSet *startset = 0;
   if (path && path[0]) startset = Find(path);
   else      startset = fWorkingDataSet;
   if (!startset) return 0;

   TDataSetIter next(startset);
   TDataSet *nextSet = 0;
   while ( (nextSet = next()) )
      if (set == nextSet) break;

   return nextSet;
}

//______________________________________________________________________________
Int_t TDataSetIter::Flag(const Char_t *path,UInt_t flag,TDataSet::EBitOpt reset)
{
   //to be documented
   TDataSet *set = Find(path);
   if (set) set->SetBit(flag,reset);
   return 0;
}
//______________________________________________________________________________
Int_t TDataSetIter::Flag(TDataSet *dataset,UInt_t flag,TDataSet::EBitOpt reset)
{
   //to be documented
   if (dataset) dataset->SetBit(flag,reset);
   return 0;
}

//______________________________________________________________________________
TDataSet *TDataSetIter::Ls(const Char_t *dirname,Option_t *opt) const {
   //
   //   Ls(const Char_t *dirname,Option_t)
   //
   //   Prints the list of the TDataSet defined with dirname
   //
   //   dirname     = 0   - prints the current dataset
   //   dirname[0]  = '/' - print TDataSet defined with dirname
   //   dirname[0] != '/' - prints DataSet with respect of the current class
   //

   TDataSet *set= 0;
   if (dirname && strlen(dirname)) set = ((TDataSetIter*)this)->Find(dirname);
   if (!set && dirname==0) set=Cwd();
   if (set) set->ls(opt);
   return set;
}

//______________________________________________________________________________
TDataSet *TDataSetIter::Ls(const Char_t *dirname,Int_t depth) const {
   //
   //   Ls(const Char_t *dirname,Int_t depth)
   //
   //   Prints the list of the TDataSet defined with dirname
   //   Returns the dataset defined by "path" or Cwd();
   //
   //   dirname     = 0   - prints the current dataset
   //   dirname[0]  = '/' - print TDataSet defined with dirname
   //   dirname[0] != '/' - prints DataSet with respect of the current class
   //
   //   depth       = 0   - print all level of the TDataSet defined with dirname
   //               > 0   - print depth levels at most of the dirname TDataSet
   //
   TDataSet *set= fWorkingDataSet;
   if (dirname && strlen(dirname)) set= ((TDataSetIter*)this)->Find(dirname);
   if (set) set->ls(depth);
   return set;
}

//______________________________________________________________________________
TDataSet *TDataSetIter::Mkdir(const Char_t *dirname)
{
   //to be documented
   TDataSet *set = 0;
   set = Find(dirname,0,kTRUE);
   if (!fNext)  Reset();  // Create a new iterator
   // If this dataset is first one then make it the root and working
   if (!fRootDataSet ) fRootDataSet  = set;
   if (!fWorkingDataSet) fWorkingDataSet = fRootDataSet;
   return set;
}

//______________________________________________________________________________
void TDataSetIter::Notify(TDataSet *)
{
   //
   //  Notify(TDataSet *dataset)
   //
   //  This dummy method is called when TDataSetIter::Find dives in "dataset"
   //  to look for thew next level of the dataset's
   //  printf("void TDataSetIter::Notify(TDataSet *) level: %d %s\n",fDepth,ds->GetName());
   //
}

//______________________________________________________________________________
TDataSet *TDataSetIter::Rmdir(TDataSet *dataset,Option_t *)
{
   //
   //  Remove the TDataSet *dataset from the current dataset
   //  If the current dataset is the deleted dataset the its parent
   //  becomes the "current dataset" or 0 if this dataset has no parent.
   //
   //  returns: the "current dataset" pointer
   //
   //
   TDataSet *set = dataset;
   if (set) {
      if (set == fWorkingDataSet) {
         fWorkingDataSet = set->GetParent();
      }
      if (set == fRootDataSet) {
         fRootDataSet = 0;
      }
      delete set;
   }
   return Cwd();
}

//______________________________________________________________________________
TDataSet *TDataSetIter::Next( TDataSet::EDataSetPass mode)
{
   ////////////////////////////////////////////////////////////////////////////////
   //
   // returns the pointer the "next" TDataSet object
   //         = 0 if all objects have been returned.
   //
   //  mode = kContinue  - default normal mode
   //         kPrune     - stop passing of the current branch but continue with the next one if any
   //         kUp        - break passing, return to the previous level, then continue
   //         all other  - are treated as "kContinue"
   //
   ////////////////////////////////////////////////////////////////////////////////

   if (fMaxDepth==1) fDataSet = fNext ? NextDataSet(*fNext) :0;
   else {
      // Check the whether the next level does exist
      if (fDepth==0) fDepth = 1;
      if (fDataSet && fDataSet != fgNullDataSet &&
         (fDepth < fMaxDepth || fMaxDepth ==0) && mode == TDataSet::kContinue )
      {
         // create the next level iterator, go deeper
         TSeqCollection *list  = fDataSet->GetCollection();
         // Look for the next level
         if (list && list->GetSize() ) {
            fDepth++;
            if (fDepth >= 100) {
               Error("Next()"
                  ," too many (%d) nested levels of your TDataSet has been detected",fDepth);
               return 0;
            }
            fNextSet[fDepth-1] = new TIter(list);
         }
      }

      // Pick the next object of the current level
      TIter *next = fNextSet[fDepth-1];
      if (next) {
         fDataSet = 0;
         if (mode != TDataSet::kUp) fDataSet = NextDataSet(*next);

         // Go upstair if the current one has been escaped
         if (!fDataSet) {
            // go backwards direction
            while (!fDataSet && fDepth > 1) {
               fDepth--;
               delete next;
               next =  fNextSet[fDepth-1];
               TDataSet *set = NextDataSet(*next);
               if (set)
                  fDataSet = set;
            }
         }
      }
   }
   return (TDataSet *)fDataSet;
}
//______________________________________________________________________________
TDataSet *TDataSetIter::NextDataSet(TIter &next)
{
   //to be documented
   TDataSet *ds = (TDataSet *)next();
   if (ds) Notify(ds);
   return ds;
}

//______________________________________________________________________________
TDataSet *TDataSetIter::NextDataSet(Int_t nDataSet)
{
   // Pick the next object of the  level provided
   TIter *next = fNextSet[nDataSet];
   if (next) return NextDataSet(*next);
   return 0;
}
//______________________________________________________________________________
TDataSet  *TDataSetIter::FindByPath(const Char_t *path, TDataSet *rootset,Bool_t mkdir)
{
   //to be documented
   return Find(path,rootset,mkdir);
}

//______________________________________________________________________________
TDataSet *TDataSetIter::Find(const Char_t *path, TDataSet *rootset,
                             Bool_t mkdirflag,Bool_t titleFlag)
{
   ////////////////////////////////////////////////////////////////////////////////
   //                                                                            //
   //     titleFlag = kFALSE; use object name as key (by default)                //
   //                 kTRUE;  use object title as key  and ignore mkdirFlag      //
   //                                                                            //
   //           "path" ::= <relative path> | <absolute path> | <empty>           //
   //                                                                            //
   //  "relative path" ::= <dataset name> | <dataset name>/<dataset name>        //
   //                                                                            //
   //  "absolute path" ::= /<relative path>                                      //
   //  "empty"         ::= zero pointer | pointer to zero length string          //
   //                                                                            //
   // "relative path": the search is done against of fWorkingDataSet data mem    //
   // "absolute path": the search is done against of fRootDataSet    data mem    //
   // "empty path"   : no search is done just next TDataSet is returned if any   //
   //                                                                            //
   //  Remark: This version can not treat any "special name" like "..", ".", etc //
   //  ------                                                                    //
   ////////////////////////////////////////////////////////////////////////////////
   TDataSet *dataset=0,*dsnext=0,*ds=0;
   Int_t len=0,nextlen=0,yes=0,anywhere=0,rootdir=0;
   const Char_t *name=0,*nextname=0;
   TSeqCollection *tl=0;

   name = path;
   if (!name) return rootset;
   dataset = rootset;
   if (!dataset) {// Starting point
      rootdir = 1999;
      dataset = (path[0]=='/') ? fRootDataSet:fWorkingDataSet;}

   if (name[0] == '/') name++;

   if (!strncmp(name,".*/",3)) {anywhere=1998; name +=3;}

   len = strcspn(name," /");
   if (!len) return dataset;

   if (!dataset) goto NOTFOUND;

   //      Check name of root directory
   if (rootdir)
   {
      nextname = titleFlag ? dataset->GetTitle() : dataset->GetName();
      nextlen  = strlen(nextname);
      if (nextlen==len && !strncmp(name,nextname,len))
         return Find(name+len,dataset,mkdirflag,titleFlag);
   }

   tl = dataset->GetCollection();
   if (tl) {
      TIter next(tl);
      while ( (dsnext = NextDataSet(next)) )
      { //horisontal loop
         nextname = titleFlag ? dataset->GetTitle() : dsnext->GetName();
         if (!nextname)  continue;
         yes = name[0]=='*';     // wildcard test
         if (!yes) {             // real     test
            nextlen  = strlen(nextname);
            yes = (len == nextlen);
            if (yes)
               yes = !strncmp(name,nextname,len);
         }

         if (yes)
         {//go down
            if (fDepth == 0) fDepth = 1;
            Notify(dsnext);
            fDepth++;
            ds = Find(name+len,dsnext,mkdirflag,titleFlag);
            fDepth--;
            if (ds)
               return ds;
         }

         if (!anywhere) continue;        // next horizontal
         ds = Find(name,dsnext,mkdirflag,titleFlag);
         if (ds)
            return ds;
      }                                  // end of while
   }

NOTFOUND:
   if (mkdirflag && !titleFlag)
   {
      // create dir the same type as the type of the fRootDataSet if present
      // Create TDataSet by default.
      char buf[512];buf[0]=0; strncat(buf,name,len);
      if (!fRootDataSet)
         ds = new TDataSet(buf);
      else {
         ds = fRootDataSet->Instance();
         ds->SetName(buf);
      }

      if (!fRootDataSet)         fRootDataSet    = ds;
      if (!fWorkingDataSet)      fWorkingDataSet = ds;
      if (dataset)
         dataset->Add(ds);
      else {
         dataset = ds;
         name +=len;
      }

      return Find(name,dataset,mkdirflag);
   }

   return 0;
}
//______________________________________________________________________________
void TDataSetIter::Reset(TDataSet *l, int depth)
{
   //
   // TDataSet *l != 0 means the new start pointer
   //    depth      != 0 means the new value for the depth
   //                    otherwise the privious one is used;
   //
   fDataSet = fgNullDataSet;
   if (fMaxDepth != 1) {
      // clean all interators
      Int_t level = fDepth;
      if (level) level--;
      for (int i = level;i>=0;i--) {
         TIter *s = fNextSet[i];
         if (s) delete s;
      }
      fNext = 0; // this iterator has been deleted in the loop above
   }

   fDepth = 0;

   if (l) {
      fRootDataSet    = l;
      fWorkingDataSet = l;
      SafeDelete(fNext);
      if (fRootDataSet->GetCollection() )
         fNext = new TIter(fRootDataSet->GetCollection() );
   }
   else {
      fWorkingDataSet = fRootDataSet;
      if (fNext)
         fNext->Reset();
      else if (fRootDataSet && fRootDataSet->GetCollection() )
         fNext = new TIter(fRootDataSet->GetCollection() );
   }
   // set the new value of the maximum depth to bypass
   if (depth) fMaxDepth = depth;
}
//______________________________________________________________________________
TDataSet *TDataSetIter::Shunt(TDataSet *set, TDataSet *dataset)
{
   ///////////////////////////////////////////////////////////////////////////////
   //                                                                           //
   // Shunt - moves the set to the dataset defined with the second parameters   //
   //                                                                           //
   // TDataSet dataset != 0 - Add the set to the TDataSet *dataset              //
   //                                                                           //
   //                     = 0 - (by default) to the current TDataSet defined    //
   //                          with fWorkingDataSet data member                 //
   //                                                                           //
   //  returns  the pointer to set if successful or ZERO pointer                //
   //  =======                                                                  //
   //                                                                           //
   //  Note: If this TDataSetIter is empty (i.e. Cwd() returns 0), the "set"    //
   //        becomes the "root" dataset of this iterator                        //                                                                         //
   ///////////////////////////////////////////////////////////////////////////////

   if (!set) return 0;
   TDataSet *s =  dataset;
   if (!s) s = Cwd();
   if (s) {
      s->Shunt(set);
      s = set;
   }
   else {
      //  make the coming dataset the current one for the iterator
      s = set;
      fRootDataSet    = s;
      fWorkingDataSet = s;
      if (fNext) {
         Error("Shunt","TDataSetIter.has been corrupted ;-!");
         delete fNext;
         fNext = 0;
      }
      fNext = new TIter(s->GetCollection() );
   }
   return s;
}

//______________________________________________________________________________
TDataSet *TDataSetIter::Shunt(TDataSet *dataset, const Char_t *path)
{
   ///////////////////////////////////////////////////////////////////////////////
   //                                                                           //
   // Shunt                                                                     //
   //                                                                           //
   // Char_t path != 0 - Move a TDataSet dataset from its parent to             //
   //                    the TDataSet dataset                                   //
   //                    defined with "path"                                    //
   //              = 0 - (by default) to the current TDataSet defined           //
   //                    with fWorkingDataSet data member                       //
   //                                                                           //
   //  returns the dataset is success or ZERO pointer                           //
   //  =======                                                                  //
   //                                                                           //
   ///////////////////////////////////////////////////////////////////////////////
   if (!dataset) return 0;
   TDataSet *set = 0;
   if (path && strlen(path)) set = Find(path);
   return Shunt(dataset,set);
}

//______________________________________________________________________________
TDataSet *TDataSetIter::operator[](const Char_t *path)
{
   //
   // operator [] returns the pointer to the TDataSet if it does contain
   // any data (TTable for example)
   //
   //  Input:
   //     path  = The path to the dataset to find
   //
   //  Output:
   //     pointer to the dataset if it found and
   //     its TDataSet::HasData() method returns non-zero
   //     (see for example TTable::HasData() )
   TDataSet *dataSet = Find(path);
   if (dataSet && dataSet->HasData()) return dataSet;
   return 0;
}
