// @(#)root/hbook:$Name:  $:$Id: THbookFile.cxx,v 1.6 2002/02/20 17:05:16 brun Exp $
// Author: Rene Brun   18/02/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////
//  This class is an interface to the Hbook objects in Hbook files
//  Any Hbook object (1-D, 2-D, Profile, RWN or CWN can be read
//  NB: a THbookFile can only be used in READ mode
//      Use the utility in $ROOTSYS/bin/h2root to convert Hbook to Root
//
// Example of use:
//  gSystem->Load("libHbook");
//  THbookFile f("myfile.hbook");
//  f.ls();
//  TH1F *h1 = (TH1F*)f.Get(1);  //import histogram ID=1 in h1
//  h1->Fit("gaus");
//  THbookTree *T = (THbookTree*)f.Get(111); //import ntuple header
//  T->Print();  //show the Hbook ntuple variables
//  T->Draw("x","y<0"); // as in normal TTree::Draw
//
//  THbookFile can be browsed via TBrowser.
//  
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "TROOT.h"
#include "THbookFile.h"
#include "TH2.h"
#include "THbookTree.h"
#include "THbookBranch.h"
#include "THbookKey.h"
#include "TGraph.h"
#include "TProfile.h"
#include "TTreeFormula.h"
#include "TLeafI.h"
#include "TBrowser.h"
#include "TSystem.h"

#define PAWC_SIZE 2000000

//  Define the names of the Fortran common blocks for the different OSs

#ifndef WIN32
#  define pawc pawc_
#  define quest quest_
#  define hcbits hcbits_
#  define hcbook hcbook_
#  define rzcl rzcl_
int pawc[PAWC_SIZE];
int quest[100];
int hcbits[37];
int hcbook[51];
int rzcl[11];
#else
#  define pawc   PAWC
#  define quest  QUEST
#  define hcbits HCBITS
#  define hcbook HCBOOK
#  define rzcl   RZCL
extern "C" int pawc[PAWC_SIZE];
extern "C" int quest[100];
extern "C" int hcbits[37];
extern "C" int hcbook[51];
extern "C" int rzcl[11];
#endif

int *iq, *lq;
float *q;
char idname[128];
int nentries;
char chtitl[128];
int ncx,ncy,nwt,idb;
int lcont, lcid, lcdir;
float xmin,xmax,ymin,ymax;
const Int_t kMIN1 = 7;
const Int_t kMAX1 = 8;

static Int_t lastentry = -1;

//  Define the names of the Fortran subroutine and functions for the different OSs

#ifndef WIN32
# define hlimit  hlimit_
# define hldir   hldir_
# define hropen  hropen_
# define hrend   hrend_
# define hrin    hrin_
# define hnoent  hnoent_
# define hgive   hgive_
# define hgiven  hgiven_
# define hgnpar  hgnpar_
# define hgnf    hgnf_
# define hgnt    hgnt_
# define hgntf   hgntf_
# define hgnt1   hgnt1_
# define rzink   rzink_
# define hdcofl  hdcofl_
# define hmaxim  hmaxim_
# define hminim  hminim_
# define hdelet  hdelet_
# define hntvar2 hntvar2_
# define hntvar3 hntvar3_
# define hbname  hbname_
# define hbnamc  hbnamc_
# define hbnam   hbnam_
# define hi      hi_
# define hie     hie_
# define hif     hif_
# define hij     hij_
# define hix     hix_
# define hijxy   hijxy_
# define hije    hije_
# define hcdir   hcdir_

# define type_of_call
# define DEFCHAR  const char*
# define PASSCHAR(string) string
#else
# define hlimit  HLIMIT
# define hldir   HLDIR
# define hropen  HROPEN
# define hrend   HREND
# define hrin    HRIN
# define hnoent  HNOENT
# define hgive   HGIVE
# define hgiven  HGIVEN
# define hgnpar  HGNPAR
# define hgnf    HGNF
# define hgnt    HGNT
# define hgntf   HGNTF
# define hgnt1   HGNT1
# define rzink   RZINK
# define hdcofl  HDCOFL
# define hmaxim  HMAXIM
# define hminim  HMINIM
# define hdelet  HDELET
# define hntvar2 HNTVAR2
# define hntvar3 HNTVAR3
# define hbname  HBNAME
# define hbnamc  HBNAMC
# define hbnam   HBNAM
# define hi      HI
# define hie     HIE
# define hif     HIF
# define hij     HIJ
# define hix     HIX
# define hijxy   HIJXY
# define hije    HIJE
# define hcdir   HCDIR
# define type_of_call  _stdcall
# define DEFCHAR  const char*, const int
# define PASSCHAR(string) string, strlen(string)
#endif

extern "C" void  type_of_call hlimit(const int&);
#ifndef WIN32
extern "C" void  type_of_call hropen(const int&,DEFCHAR,DEFCHAR,DEFCHAR,
                        const int&,const int&,const int,const int,const int);
extern "C" void  type_of_call hrend(DEFCHAR,const int);
#else
extern "C" void  type_of_call hropen(const int&,DEFCHAR,DEFCHAR,DEFCHAR,
                        const int&,const int&);
extern "C" void  type_of_call hrend(DEFCHAR);
#endif

extern "C" void  type_of_call hrin(const int&,const int&,const int&);
extern "C" void  type_of_call hnoent(const int&,const int&);
#ifndef WIN32
extern "C" void  type_of_call hgive(const int&,DEFCHAR,const int&,const float&,const float&,
   const int&,const float&,const float&,const int&,const int&,const int);
#else
extern "C" void  type_of_call hgive(const int&,DEFCHAR,const int&,const float&,const float&,
   const int&,const float&,const float&,const int&,const int&);
#endif

  //SUBROUTINE HGNT1(IDD,BLKNA1,VAR,IOFFST,NVAR,IDNEVT,IERROR)
#ifndef WIN32
extern "C" void  type_of_call hgiven(const int&,DEFCHAR,const int&,DEFCHAR,
   const float&,const float&,const int,const int);
extern "C" void  type_of_call hgnt1(const int&,DEFCHAR,DEFCHAR,const int&,const int&,const int&,const int&,const int,const int);
#else
extern "C" void  type_of_call hgiven(const int&,DEFCHAR,const int&,DEFCHAR,
   const float&,const float&);
extern "C" void  type_of_call hgnt1(const int&,DEFCHAR,DEFCHAR,const int&,const int&,const int&,const int&);
#endif

#ifndef WIN32
extern "C" void  type_of_call hntvar2(const int&,const int&,DEFCHAR,DEFCHAR,DEFCHAR,int&,int&,int&,int&,const int,const int, const int);
extern "C" void  type_of_call hntvar3(const int&,const int&,DEFCHAR, const int);
#else
extern "C" void  type_of_call hntvar2(const int&,const int&,DEFCHAR,DEFCHAR,DEFCHAR,int&,int&,int&,int&);
extern "C" void  type_of_call hntvar3(const int&,const int&,DEFCHAR);
#endif

#ifndef WIN32
extern "C" void  type_of_call hbnam(const int&,DEFCHAR,const int&,DEFCHAR,const int&,const int, const int);
#else
extern "C" void  type_of_call hbnam(const int&,DEFCHAR,const int&,DEFCHAR,const int&);
#endif

extern "C" void  type_of_call hgnpar(const int&,const char *,const int);
extern "C" void  type_of_call hgnf(const int&,const int&,const float&,const int&);
extern "C" void  type_of_call hgnt(const int&,const int&,const int&);
extern "C" void  type_of_call hgntf(const int&,const int&,const int&);
extern "C" void  type_of_call rzink(const int&,const int&,const char *,const int);
extern "C" void  type_of_call hdcofl();
extern "C" void  type_of_call hmaxim(const int&,const float&);
extern "C" void  type_of_call hminim(const int&,const float&);
extern "C" void  type_of_call hdelet(const int&);
extern "C" float type_of_call hi(const int&,const int&);
extern "C" float type_of_call hie(const int&,const int&);
extern "C" float type_of_call hif(const int&,const int&);
extern "C" float type_of_call hij(const int&,const int&,const int&);
extern "C" void  type_of_call hix(const int&,const int&,const float&);
extern "C" void  type_of_call hijxy(const int&,const int&,const int&,const float&,const float&);
extern "C" float type_of_call hije(const int&,const int&,const int&);
#ifndef WIN32
extern "C" void  type_of_call hcdir(DEFCHAR,DEFCHAR ,const int,const int);
extern "C" void  type_of_call hldir(DEFCHAR,DEFCHAR ,const int,const int);
#else
extern "C" void  type_of_call hcdir(DEFCHAR,DEFCHAR);
extern "C" void  type_of_call hldir(DEFCHAR,DEFCHAR);
#endif

Bool_t THbookFile::fgPawInit = kFALSE;
Int_t  *THbookFile::fgLuns   = 0;
R__EXTERN TTree *gTree;

ClassImp(THbookFile)

//______________________________________________________________________________
THbookFile::THbookFile() : TNamed()
{
   fList = new TList();
   fKeys = new TList();
}

//1_____________________________________________________________________________
THbookFile::THbookFile(const char *fname, Int_t lrecl)
           :TNamed(fname,"")
{
//  Constructor for an HBook file object

  // Initialize the Hbook/Zebra store
   Int_t i;
   if (!fgPawInit) {
     fgPawInit = kTRUE;
     lq = &pawc[9];
     iq = &pawc[17];
     void *qq = iq;
     q = (float*)qq;
     int pawc_size = PAWC_SIZE;
     hlimit(pawc_size);
     fgLuns = new Int_t[10];
     for (i=0;i<10;i++) fgLuns[i] = 0;
  }

  //find a free logical unit (max 10)
  fLun = 0;
  for (i=0;i<10;i++) {
     if (fgLuns[i] == 0) {
        fLun = 10+i;
        fgLuns[i] = 1;
        break;
     }
  }
  if (fLun == 0) {
     Error("THbookFile","Too many HbookFiles\n");
     return;
  }
  char topdir[20];
  sprintf(topdir,"lun%d",fLun);
  
  Int_t ier;
#ifndef WIN32
  hropen(fLun,PASSCHAR(topdir),PASSCHAR(fname),PASSCHAR("p"),lrecl,ier,strlen(topdir),strlen(fname),1);
#else
  hropen(fLun,PASSCHAR(topdir),PASSCHAR(fname),PASSCHAR("p"),lrecl,ier);
#endif
  sprintf(topdir,"//lun%d",fLun);
  SetTitle(topdir);
  fCurDir = topdir;
  
  if (ier) printf (" Error on hropen was %d \n", ier);
  if (quest[0]) {
     printf("Error cannot open input file: %s\n",fname);
  }
  
  gROOT->GetListOfBrowsables()->Add(this,fname);

  fList = new TList();
  fKeys = new TList();
  for (Int_t key=1;key<1000000;key++) {
     int z0 = 0;
     rzink(key,z0,"S",1);
     if (quest[0]) break;
     if (quest[13] & 8) continue;
     Int_t id = quest[20];
     THbookKey *akey = new THbookKey(id,this);
     fKeys->Add(akey);
  }
}

//______________________________________________________________________________
THbookFile::~THbookFile()
{

   Close(); 
   delete fList;
   delete fKeys;
}

//______________________________________________________________________________
void THbookFile::Browse(TBrowser *b)
{
// to be implemented

   if( b ) {
        b->Add(fList, "memory");
        b->Add(fKeys, "IDs on disk");
   } 
   cd();  
}

//______________________________________________________________________________
Bool_t THbookFile::cd(const char *dirname)
{
// change directory to dirname

   Int_t nch = strlen(dirname);
   if (nch == 0) {
#ifndef WIN32
       hcdir(PASSCHAR(fCurDir.Data()),PASSCHAR(" "),fCurDir.Length(),1);
#else
       hcdir(PASSCHAR(fCurDir.Data()),PASSCHAR(" "));
#endif
       return kTRUE;      
   }
   
   char cdir[512];
   Int_t i;
   for (i=0;i<512;i++) cdir[i] = ' ';
   cdir[511] = 0;
#ifndef WIN32
  hcdir(PASSCHAR(dirname),PASSCHAR(" "),nch,1);
  hcdir(PASSCHAR(cdir),PASSCHAR("R"),511,1);
#else
  hcdir(PASSCHAR(dirname),PASSCHAR(" "));
  hcdir(PASSCHAR(cdir),PASSCHAR("R"));
#endif
  for (i=510;i>=0;i--) {
     if (cdir[i] != ' ') break;
     cdir[i] = 0;
  }  
  fCurDir = cdir; 
  printf("fCurdir=%s\n",fCurDir.Data());
     
  return kTRUE;
}

//______________________________________________________________________________
void THbookFile::Close(Option_t *)
{
// Close the Hbook file
   
   cd();
   
   fList->Delete();
   fKeys->Delete();
   fgLuns[fLun-10] = 0;
   hdelet(0);
#ifndef WIN32
  hrend(PASSCHAR(GetTitle()),strlen(GetTitle()));
#else
  hrend(PASSCHAR(GetTitle()));
#endif
}

//______________________________________________________________________________
void THbookFile::DeleteID(Int_t id)
{
   hdelet(id);
}

//______________________________________________________________________________
TObject *THbookFile::FindObject(const char *name) const
{
// return object with name in fList in memory
   return fList->FindObject(name);
}

//______________________________________________________________________________
TObject *THbookFile::FindObject(const TObject *obj) const
{
// return object with pointer obj in fList in memory
   return fList->FindObject(obj);
}

//______________________________________________________________________________
TObject *THbookFile::Get(Int_t idd)
{
// import Hbook object with identifier idd in memory

  Int_t id = 0;
  for (Int_t key=1;key<1000000;key++) {
     int z0 = 0;
     rzink(key,z0,"S",1);
     if (quest[0]) break;
     if (quest[13] & 8)  continue;
     id = quest[20];
     if (id == idd) break;
  }
  if (id == 0) return 0;

  int i999 = 999;
  hrin(id,i999,0);
  if (quest[0]) {
     printf("Error cannot read ID = %d\n",id);
     return 0;
  }
  hdcofl();
  lcid  = hcbook[10];
  lcont = lq[lcid-1];
  TObject *obj = 0;
  if (hcbits[3]) {
     if (iq[lcid-2] == 2) obj = ConvertRWN(id);
     else                 obj = ConvertCWN(id);
     //hdelet(id); //cannot be deleted here since used in GetEntry
     if (obj) fList->Add(obj);
     return obj;
  }
  if (hcbits[0] && hcbits[7]) {
     obj = ConvertProfile(id);
     hdelet(id);
     if (obj) fList->Add(obj);
     return obj;
  }
  if (hcbits[0]) {
     obj = Convert1D(id);
     hdelet(id);
     if (obj) fList->Add(obj);
     return obj;
  }
  if (hcbits[1] || hcbits[2]) {
     obj = Convert2D(id);
     hdelet(id);
     if (obj) fList->Add(obj);
     return obj;
  }
  return obj;
}


//______________________________________________________________________________
Int_t THbookFile::GetEntry(Int_t entry, Int_t id, Int_t atype, Float_t *x)
{
// Read in memory all columns of entry number of ntuple id from the Hbook file

   Int_t ier = 0;
   if (atype == 0) {
      hgnf(id,entry+1,x[0],ier);
   } else {
      hgnt(id,entry+1,ier);
   }
   return 0;
}

//______________________________________________________________________________
Int_t THbookFile::GetEntryBranch(Int_t entry, Int_t id)
{
// Read in memory only the branch bname

   if (entry == lastentry) return 0;
   lastentry = entry;
   Int_t ier = 0;
   //uses the fast read method using the Hbook tables computed in InitLeaves
   hgntf(id,entry+1,ier);
   //old alternative slow method
//#ifndef WIN32
//   hgnt1(id,PASSCHAR(blockname),PASSCHAR(branchname),0,-1,entry+1,ier,strlen(blockname),strlen(branchname));
//#else
//   hgnt1(id,PASSCHAR(blockname),PASSCHAR(branchname),0,-1,entry+1,ier);
//#endif
   return 0;
}


//______________________________________________________________________________
void THbookFile::InitLeaves(Int_t id, Int_t var, TTreeFormula *formula)
{
// This function is called from the first entry in TTreePlayer::InitLoop 
// It analyzes the list of variables involved in the current query
// and pre-process the internal Hbook tables to speed-up the search
// at the next entries.
   
  if (!formula) return;
   Int_t ncodes = formula->GetNcodes();
   for (Int_t i=1;i<=ncodes;i++) {
      TLeaf *leaf = formula->GetLeaf(i-1);
      if (!leaf) continue;
      if (var == 5) {
         //leafcount may be null in case of a fix size array
         if (leaf->GetLeafCount()) leaf = leaf->GetLeafCount();
      }
      Int_t last = 0;
      if (var == 1 && i == ncodes) last = 1;
#ifndef WIN32
      hntvar3(id,last,PASSCHAR(leaf->GetName()),strlen(leaf->GetName()));
#else
      hntvar3(id,last,PASSCHAR(leaf->GetName()));
#endif
   }
}


//______________________________________________________________________________
TFile *THbookFile::Convert2root(const char *rootname, Int_t lrecl, Option_t *option)
{
// Convert this Hbook file to a Root file with name rootname.
// if rootname="', rootname = hbook file name with .root instead of .hbook
// By default, the Root file is connected and returned 
// option:
//       - "NO" do not connect the Root file
//       - "C"  do not compress file (default is to compress)
//       - "L"  do not convert names to lower case (default is to convert)
   
   TString opt = option;
   opt.ToLower();

   char cmd[512];
   char rfile[512];
   Int_t nch = strlen(rootname);
   if (nch) {
      strcpy(rfile,rootname);
   } else {
      strcpy(rfile,GetName());
      char *dot = strrchr(rfile,'.');
      if (dot) strcpy(dot+1,"root");
      else     strcat(rfile,".root");
   }

   sprintf(cmd,"h2root %s %s",GetName(),rfile);
   if (opt.Contains("c")) strcat (cmd," 0");
   if (opt.Contains("l")) strcat (cmd," 0");
   
   gSystem->Exec(cmd);
   
   if (opt.Contains("no")) return 0;
   TFile *f = new TFile(rfile);
   if (f->IsZombie()) {delete f; f = 0;}
   return f;
}      


//______________________________________________________________________________
TObject *THbookFile::ConvertCWN(Int_t id)
{
// Convert the Column-Wise-Ntuple id to a Root Tree

  const int Nchar=9;
  int nvar;
  int i,j;
  int nsub,itype,isize,ielem;
  char *chtag_out;
  float rmin[1000], rmax[1000];

  if (id > 0) sprintf(idname,"h%d",id);
  else        sprintf(idname,"h_%d",-id);
  hnoent(id,nentries);
  //printf(" Converting CWN with ID= %d, nentries = %d\n",id,nentries);
  nvar=0;
#ifndef WIN32
  hgiven(id,chtitl,nvar,PASSCHAR(""),rmin[0],rmax[0],80,0);
#else
  hgiven(id,chtitl,80,nvar,PASSCHAR(""),rmin[0],rmax[0]);
#endif
  chtag_out = new char[nvar*Nchar+1];
  Int_t *charflag = new Int_t[nvar];
  Int_t *lenchar  = new Int_t[nvar];
  Int_t *boolflag = new Int_t[nvar];
  Int_t *lenbool  = new Int_t[nvar];
  UChar_t *boolarr = new UChar_t[10000];

  chtag_out[nvar*Nchar]=0;
  for (i=0;i<80;i++)chtitl[i]=0;
#ifndef WIN32
  hgiven(id,chtitl,nvar,chtag_out,rmin[0],rmax[0],80,Nchar);
#else
  hgiven(id,chtitl,80,nvar,chtag_out,Nchar,rmin[0],rmax[0]);
#endif

  Int_t bufpos = 0;
  Int_t isachar = 0;
  Int_t isabool = 0;
  char fullname[64];
  char name[32];
  char block[32];
  char oldblock[32];
  strcpy(oldblock,"OLDBLOCK");
  Int_t oldischar = -1;
  for (i=80;i>0;i--) {if (chtitl[i] == ' ') chtitl[i] = 0; }
  THbookTree *tree = new THbookTree(idname,id);
  tree->SetHbookFile(this);
  tree->SetType(1);

  static char *bigbuf = tree->MakeX(500000);

  gTree = tree;
#ifndef WIN32
  hbnam(id,PASSCHAR(" "),bigbuf[0],PASSCHAR("$CLEAR"),0,1,6);
#else
  hbnam(id,PASSCHAR(" "),bigbuf[0],PASSCHAR("$CLEAR"),0);
#endif

  Int_t golower  = 1;
  for(i=0; i<nvar;i++) {
     memset(name,' ',sizeof(name));
     name[sizeof(name)-1] = 0;
     memset(block,' ',sizeof(block));
     block[sizeof(block)-1] = 0;
     memset(fullname,' ',sizeof(fullname));
     fullname[sizeof(fullname)-1]=0;
#ifndef WIN32
     hntvar2(id,i+1,PASSCHAR(name),PASSCHAR(fullname),PASSCHAR(block),nsub,itype,isize,ielem,32,64,32);
#else
     hntvar2(id,i+1,PASSCHAR(name),PASSCHAR(fullname),PASSCHAR(block),nsub,itype,isize,ielem);
#endif
     TString HbookName = name;

     for (j=30;j>0;j--) {
        if(golower) name[j] = tolower(name[j]);
        if (name[j] == ' ') name[j] = 0;
     }
     if (golower == 2) name[0] = tolower(name[0]);

     for (j=62;j>0;j--) {
        if(golower && fullname[j-1] != '[') fullname[j] = tolower(fullname[j]);
        // convert also character after [, if golower == 2
        if (golower == 2) fullname[j] = tolower(fullname[j]);
        if (fullname[j] == ' ') fullname[j] = 0;
     }
     // convert also first character, if golower == 2
     if (golower == 2) fullname[0] = tolower(fullname[0]);
     for (j=30;j>0;j--) {
        if (block[j] == ' ') block[j] = 0;
        else break;
     }
     if (itype == 1 && isize == 4) strcat(fullname,"/F");
     if (itype == 1 && isize == 8) strcat(fullname,"/D");
     if (itype == 2) strcat(fullname,"/I");
     if (itype == 3) strcat(fullname,"/i");
//     if (itype == 4) strcat(fullname,"/i");
     if (itype == 4) strcat(fullname,"/b");
     if (itype == 5) strcat(fullname,"/C");
//printf("Creating branch:%s, block:%s, fullname:%s, nsub=%d, itype=%d, isize=%d, ielem=%d, bufpos=%d\n",name,block,fullname,nsub,itype,isize,ielem,bufpos);
     Int_t ischar;
     if (itype == 5) ischar = 1;
     else            ischar = 0;

     if (ischar != oldischar || strcmp(oldblock,block) != 0) {
        strcpy(oldblock,block);
        oldischar = ischar;
        Long_t add= (Long_t)&bigbuf[bufpos];
        Int_t lblock   = strlen(block);
#ifndef WIN32
        hbnam(id,PASSCHAR(block),add,PASSCHAR("$SET"),ischar,lblock,4);
#else
        hbnam(id,PASSCHAR(block),add,PASSCHAR("$SET"),ischar);
#endif

     }

     Int_t bufsize = 8000;
     THbookBranch *branch = new THbookBranch(name,(void*)&bigbuf[bufpos],fullname,bufsize);
     tree->GetListOfBranches()->Add(branch);
     branch->SetBlockName(block);

     //NB: the information about isachar should be saved in the branch
     // to be done
     boolflag[i] = -10;
     charflag[i] = 0;
     if (itype == 4) {isabool++; boolflag[i] = bufpos; lenbool[i] = ielem;}
     bufpos += isize*ielem;
     if (ischar) {isachar++; charflag[i] = bufpos-1; lenchar[i] = isize*ielem;}
     TObjArray *ll= branch->GetListOfLeaves();
     TLeaf *leaf = (TLeaf*)ll->UncheckedAt(0);
     if (!leaf) continue;
     TLeafI *leafcount = (TLeafI*)leaf->GetLeafCount();
     if (leafcount) {
        if (leafcount->GetMaximum() <= 0) leafcount->SetMaximum(ielem);
     }
  }
  tree->SetEntries(nentries);
  delete [] charflag;
  delete [] lenchar;
  delete [] boolflag;
  delete [] lenbool;
  delete [] boolarr;
  delete [] chtag_out;

     return tree;
}

//______________________________________________________________________________
TObject *THbookFile::ConvertRWN(Int_t id)
{
// Convert the Row-Wise-Ntuple id to a Root Tree

  const int Nchar=9;
  int nvar;
  int i,j;
  char *chtag_out;
  float rmin[1000], rmax[1000];

  if (id > 0) sprintf(idname,"h%d",id);
  else        sprintf(idname,"h_%d",-id);
  hnoent(id,nentries);
  //printf(" Converting RWN with ID= %d, nentries = %d\n",id,nentries);
  nvar=0;
#ifndef WIN32
  hgiven(id,chtitl,nvar,PASSCHAR(""),rmin[0],rmax[0],80,0);
#else
  hgiven(id,chtitl,80,nvar,PASSCHAR(""),rmin[0],rmax[0]);
#endif

  chtag_out = new char[nvar*Nchar+1];

  Int_t golower  = 1;
  chtag_out[nvar*Nchar]=0;
  for (i=0;i<80;i++)chtitl[i]=0;
#ifndef WIN32
  hgiven(id,chtitl,nvar,chtag_out,rmin[0],rmax[0],80,Nchar);
#else
  hgiven(id,chtitl,80,nvar,chtag_out,Nchar,rmin[0],rmax[0]);
#endif
  hgnpar(id,"?",1);
  char *name = chtag_out;
  for (i=80;i>0;i--) {if (chtitl[i] == ' ') chtitl[i] = 0; }
  THbookTree *tree = new THbookTree(idname,id);
  tree->SetHbookFile(this);
  tree->SetType(0);
  gTree = tree;
  Float_t *x = (Float_t*)tree->MakeX(nvar*4);

  Int_t first,last;
  for(i=0; i<nvar;i++) {
    name[Nchar-1] = 0;
    first = last = 0;
    TString HbookName = name;
    // suppress trailing blanks
    for (j=Nchar-2;j>0;j--) {
       if(golower) name[j] = tolower(name[j]);
       if (name[j] == ' ' && last == 0) name[j] = 0;
       else last = j;
    }
    if (golower == 2) name[0] = tolower(name[0]);

    // suppress heading blanks
    for (j=0;j<Nchar;j++) {
       if (name[j] != ' ') break;
       first = j+1;
    }
    Int_t bufsize = 8000;
    //tree->Branch(&name[first],&x[i],&name[first],bufsize);
    THbookBranch *branch = new THbookBranch(&name[first],&x[4*i],&name[first],bufsize);
    branch->SetAddress(&x[i]);
    branch->SetBlockName(HbookName.Data());
    tree->GetListOfBranches()->Add(branch);
    name += Nchar;
  }
  tree->SetEntries(nentries);
  return tree;
}

//______________________________________________________________________________
TObject *THbookFile::ConvertProfile(Int_t id)
{
// Convert an Hbook profile histogram into a Root TProfile
//
// the following structure is used in Hbook
//    lcid points to the profile in array iq
//    lcont = lq(lcid-1)
//    lw    = lq(lcont)
//    ln    = lq(lw)
//      if option S jbyt(iq(lw),1,2) = 1
//      if option I jbyt(iq(lw),1,2) = 2

  if (id > 0) sprintf(idname,"h%d",id);
  else        sprintf(idname,"h_%d",-id);
  hnoent(id,nentries);
  Int_t lw = lq[lcont];
  Int_t ln = lq[lw];
#ifndef WIN32
  hgive(id,chtitl,ncx,xmin,xmax,ncy,ymin,ymax,nwt,idb,80);
#else
  hgive(id,chtitl,80,ncx,xmin,xmax,ncy,ymin,ymax,nwt,idb);
#endif
  Float_t offsetx = 0.5*(xmax-xmin)/ncx;
  chtitl[4*nwt] = 0;
  const char *option= " ";
  if (iq[lw] == 1) option = "S";
  if (iq[lw] == 2) option = "I";
  TProfile *p = new TProfile(idname,chtitl,ncx,xmin,xmax,ymin,ymax,option);

  const Int_t kCON1 = 9;
  Int_t i;
  Float_t x;
  Float_t y = 0.5*(ymin+ymax);
  for (i=1;i<=ncx;i++) {
     Int_t n = Int_t(q[ln+i]);
     hix(id,i,x);
     for (Int_t j=0;j<n;j++) {
        p->Fill(x+offsetx,y);
     }
     Float_t content = q[lcont+kCON1+i];
     Float_t error   = TMath::Sqrt(q[lw+i]);
     p->SetBinContent(i,content);
     p->SetBinError(i,error);
  }
  p->SetEntries(nentries);
  return p;
}

//______________________________________________________________________________
TObject *THbookFile::Convert1D(Int_t id)
{
// Convert an Hbook 1-d histogram into a Root TH1F

  if (id > 0) sprintf(idname,"h%d",id);
  else        sprintf(idname,"h_%d",-id);
  hnoent(id,nentries);
#ifndef WIN32
  hgive(id,chtitl,ncx,xmin,xmax,ncy,ymin,ymax,nwt,idb,80);
#else
  hgive(id,chtitl,80,ncx,xmin,xmax,ncy,ymin,ymax,nwt,idb);
#endif
  chtitl[4*nwt] = 0;
  TH1F *h1;
  Int_t i;
  if (hcbits[5]) {
     Int_t lbins = lq[lcid-2];
     Double_t *xbins = new Double_t[ncx+1];
     for (i=0;i<=ncx;i++) xbins[i] = q[lbins+i+1];
     h1 = new TH1F(idname,chtitl,ncx,xbins);
     delete [] xbins;
  } else {
     h1 = new TH1F(idname,chtitl,ncx,xmin,xmax);
  }
  if (hcbits[8]) h1->Sumw2();
  TGraph *gr = 0;
  if (hcbits[11]) {
     gr = new TGraph(ncx);
     h1->GetListOfFunctions()->Add(gr);
  }

  Float_t x;
  for (i=0;i<=ncx+1;i++) {
     x = h1->GetBinCenter(i);
     h1->Fill(x,hi(id,i));
     if (hcbits[8]) h1->SetBinError(i,hie(id,i));
     if (gr && i>0 && i<=ncx) gr->SetPoint(i,x,hif(id,i));
  }
  Float_t ymin, ymax;
  if (hcbits[19]) {
     ymax = q[lcid+kMAX1];
     h1->SetMaximum(ymax);
  }
  if (hcbits[20]) {
     ymin = q[lcid+kMIN1];
     h1->SetMinimum(ymin);
  }
  h1->SetEntries(nentries);
  return h1;
}

//______________________________________________________________________________
TObject *THbookFile::Convert2D(Int_t id)
{
// Convert an Hbook 2-d histogram into a Root TH2F

  if (id > 0) sprintf(idname,"h%d",id);
  else        sprintf(idname,"h_%d",-id);
  hnoent(id,nentries);
#ifndef WIN32
  hgive(id,chtitl,ncx,xmin,xmax,ncy,ymin,ymax,nwt,idb,80);
#else
  hgive(id,chtitl,80,ncx,xmin,xmax,ncy,ymin,ymax,nwt,idb);
#endif
  chtitl[4*nwt] = 0;
  TH2F *h2 = new TH2F(idname,chtitl,ncx,xmin,xmax,ncy,ymin,ymax);
  Float_t offsetx = 0.5*(xmax-xmin)/ncx;
  Float_t offsety = 0.5*(ymax-ymin)/ncy;
  Int_t lw = lq[lcont];
  if (lw) h2->Sumw2();

  Float_t x,y;
  for (Int_t j=0;j<=ncy+1;j++) {
     for (Int_t i=0;i<=ncx+1;i++) {
        hijxy(id,i,j,x,y);
        h2->Fill(x+offsetx,y+offsety,hij(id,i,j));
        if (lw) {
           Double_t err2 = hije(id,i,j);
           h2->SetCellError(i,j,err2);
        }
     }
  }
  h2->SetEntries(nentries);
  return h2;
}

//______________________________________________________________________________
void THbookFile::ls(const char *path) const
{
// List contents of Hbook directory

   Int_t nch = strlen(path);
   if (nch == 0) {
#ifndef WIN32
       hldir(PASSCHAR(fCurDir.Data()),PASSCHAR("T"),fCurDir.Length(),1);
#else
       hldir(PASSCHAR(fCurDir.Data()),PASSCHAR("T"));
#endif
       return;
   }
   
#ifndef WIN32
  hldir(PASSCHAR(path),PASSCHAR("T"),strlen(path),1);
#else
  hldir(PASSCHAR(path),PASSCHAR("T"));
#endif
}
