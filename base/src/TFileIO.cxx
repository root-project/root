// @(#)root/base:$Name:  $:$Id: TFileIO.cxx,v 1.17 2006/07/09 05:27:53 brun Exp $
// Author: Rene Brun   24/01/2007
/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      
// TFileIO                                                       
//                                                                     
// TFileIO is the concrete implementation of TVirtualIO.
//                                     
//////////////////////////////////////////////////////////////////////////

#include "TFileIO.h"
#include "TROOT.h"
#include "TClass.h"
#include "TFile.h"
#include "TKey.h"
#include "TProcessID.h"
#include "TRefTable.h"
#include "TBufferFile.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TSystem.h"

ClassImp(TFileIO)

//______________________________________________________________________________
TFileIO::TFileIO()
{
   // Default constructor.
}

//______________________________________________________________________________
TFileIO::TFileIO(const TFileIO& io) : TVirtualIO(io) 
{ 
   //copy constructor
}

//______________________________________________________________________________
TFileIO& TFileIO::operator=(const TFileIO& io)
{
   //assignment operator
   if(this!=&io) {
      TVirtualIO::operator=(io);
   } 
   return *this;
}

//______________________________________________________________________________
TFileIO::~TFileIO()
{
   // destructor

}


//______________________________________________________________________________
TObject *TFileIO::CloneObject(const TObject *obj)
{
   // Make a clone of an object using the Streamer facility.
   // If the object derives from TNamed, this function is called
   // by TNamed::Clone. TNamed::Clone uses the optional argument newname to set
   // a new name to the newly created object.

   // if no default ctor return immediately (error issued by New())
   TObject *newobj = (TObject *)obj->IsA()->New();
   if (!newobj) return 0;

   //create a buffer where the object will be streamed
   TFile *filsav = gFile;
   gFile = 0;
   const Int_t bufsize = 10000;
   TBuffer *buffer = new TBufferFile(TBuffer::kWrite,bufsize);
   buffer->MapObject(obj);  //register obj in map to handle self reference
   ((TObject*)obj)->Streamer(*buffer);

   // read new object from buffer
   buffer->SetReadMode();
   buffer->ResetMap();
   buffer->SetBufferOffset(0);
   buffer->MapObject(newobj);  //register obj in map to handle self reference
   newobj->Streamer(*buffer);
   newobj->ResetBit(kIsReferenced);
   newobj->ResetBit(kCanDelete);
   gFile = filsav;

   delete buffer;
   return newobj;
}


//______________________________________________________________________________
TObject *TFileIO::FindObjectAny(const char *name) const
{
   // Scan the memory lists of all files for an object with name
   
   TFile *f;
   TIter next(gROOT->GetListOfFiles());
   while ((f = (TFile*)next())) {
      TObject *obj = f->GetList()->FindObject(name);
      if (obj) return obj;
   }
   return 0;
}

//______________________________________________________________________________
TProcessID *TFileIO::GetLastProcessID(TBuffer &R__b, TRefTable *reftable) const
{
   // Return the last TProcessID in the file
   
   TFile *file = (TFile*)(R__b.GetParent());
   // warn if the file contains > 1 PID (i.e. if we might have ambiguity)
   if (file && !reftable->TestBit(TRefTable::kHaveWarnedReadingOld) && file->GetNProcessIDs()>1) {
      Warning("ReadBuffer", "The file was written during several processes with an "
         "older ROOT version; the TRefTable entries might be inconsistent.");
      reftable->SetBit(TRefTable::kHaveWarnedReadingOld);
   }

   // the file's last PID is the relevant one, all others might have their tables overwritten
   TProcessID *fileProcessID = TProcessID::GetProcessID(0);
   if (file && file->GetNProcessIDs() > 0) {
      // take the last loaded PID
      fileProcessID = (TProcessID *) file->GetListOfProcessIDs()->Last();
   }
   return fileProcessID;
}

//______________________________________________________________________________
UInt_t TFileIO::GetTRefExecId()
{
   // Return the exec id stored in the current TStreamerInfo element
   
   return TStreamerInfo::GetCurrentElement()->GetUniqueID();
}   

//______________________________________________________________________________
TObject *TFileIO::Open(const char *name, Option_t *option,const char *ftitle, Int_t compress, Int_t netopt)
{
   // Interface to TFile::Open
   
   return TFile::Open(name,option,ftitle,compress,netopt);

}


//______________________________________________________________________________
Int_t TFileIO::ReadObject(TObject *obj, const char *keyname)
{
   // Read object with keyname from the current directory
   // Read contents of object with specified name from the current directory.
   // First the key with keyname is searched in the current directory,
   // next the key buffer is deserialized into the object.
   // The object must have been created before via the default constructor.
   // See TObject::Write().

   if (!gFile) { Error("Read","No file open"); return 0; }
   TKey *key = (TKey*)gDirectory->GetListOfKeys()->FindObject(keyname);
   if (!key)   { Error("Read","Key not found"); return 0; }
   return key->Read(obj);
}

//______________________________________________________________________________
UShort_t TFileIO::ReadProcessID(TBuffer &R__b, TProcessID *pid)
{
   //The TProcessID with number pidf is read from file. (static function)
   //If the object is not already entered in the gROOT list, it is added.

   UShort_t pidf;
   R__b >> pidf;
   pidf += R__b.GetPidOffset();
   TFile *file = (TFile*)R__b.GetParent();
   pid = 0;
   if (!file) {
      if (!pidf) pid = TProcessID::GetPID(); //may happen when cloning an object
      return pidf;
   }
   TObjArray *pids = file->GetListOfProcessIDs();
   if (pidf < pids->GetSize()) pid = (TProcessID *)pids->UncheckedAt(pidf);
   if (pid) {
      pid->CheckInit();
      return pidf;
   }

   //check if fProcessIDs[uid] is set in file
   //if not set, read the process uid from file
   char pidname[32];
   sprintf(pidname,"ProcessID%d",pidf);
   TDirectory *dirsav = gDirectory;
   file->cd();
   pid = (TProcessID *)file->Get(pidname);
   if (dirsav) dirsav->cd();
   if (gDebug > 0) {
      printf("ReadProcessID, name=%s, file=%s, pid=%lx\n",pidname,file->GetName(),(Long_t)pid);
   }
   if (!pid) {
      //file->Error("ReadProcessID","Cannot find %s in file %s",pidname,file->GetName());
      return pidf;
   }
      //check that a similar pid is not already registered in fgPIDs
   TObjArray *pidslist = TProcessID::GetPIDs();
   TIter next(pidslist);
   TProcessID *p;
   while ((p = (TProcessID*)next())) {
      if (!strcmp(p->GetTitle(),pid->GetTitle())) {
         delete pid;
         pids->AddAtAndExpand(p,pidf);
         p->IncrementCount();
         pid = p;
         return pidf;
      }
   }
   pids->AddAtAndExpand(pid,pidf);
   pid->IncrementCount();
   pidslist->Add(pid);
   Int_t ind = pidslist->IndexOf(pid);
   pid->SetUniqueID((UInt_t)ind);
   return pidf;
}


//______________________________________________________________________________
void TFileIO::ReadRefUniqueID(TBuffer &R__b, TObject *obj)
{
   //if the object is referenced, we must read its old address
   //and store it in the ProcessID map in gROOT
   
   TProcessID *pid = 0;
   //UShort_t pidf = TProcessID::ReadProcessID(R__b, pid);
   ReadProcessID(R__b, pid);
   if (pid) {
      UInt_t uid  = obj->GetUniqueID();
      UInt_t gpid = pid->GetUniqueID();
      if (gpid>=0xff) {
         uid = uid | 0xff000000;
      } else {
         uid = ( uid & 0xffffff) + (gpid<<24);
      }
      obj->SetUniqueID(uid);
      pid->PutObjectWithID(obj);
   }
}

//______________________________________________________________________________
Int_t TFileIO::SaveObjectAs(const TObject *obj, const char *filename, Option_t *option)
{
   // Save object in filename (static function)
   // if filename is null or "", a file with "objectname.root" is created.
   // The name of the key is the object name.
   // If the operation is successful, it returns the number of bytes written to the file
   // otherwise it returns 0.
   // By default a message is printed. Use option "q" to not print the message.
   
   if (!obj) return 0;
   TDirectory *dirsav = gDirectory;
   TString fname = filename;
   if (!filename || strlen(filename) == 0) {
      fname = Form("%s.root",obj->GetName());
   }
   TFile *local = TFile::Open(fname.Data(),"recreate");
   if (!local) return 0;
   Int_t nbytes = obj->Write();
   delete local;
   if (dirsav) dirsav->cd();
   TString opt = option;
   opt.ToLower();
   if (!opt.Contains("q")) {
      if (!gSystem->AccessPathName(fname.Data())) obj->Info("SaveAs", "ROOT file %s has been created", fname.Data());
   }
   return nbytes;
}

//______________________________________________________________________________
void TFileIO::SetRefAction(TObject *ref, TObject *parent)
{
   // Find the action to be executed in the dictionary of the parent class
   // and store the corresponding exec number into fBits.
   // This function searches a data member in the class of parent with an
   // offset corresponding to this.
   // If a comment "TEXEC:" is found in the comment field of the data member,
   // the function stores the exec identifier of the exec statement
   // following this keyword.
   
   Int_t offset = (char*)ref - (char*)parent;
   TClass *cl = parent->IsA();
   cl->BuildRealData(parent);
   TStreamerInfo *info = cl->GetStreamerInfo();
   TIter next(info->GetElements());
   TStreamerElement *element;
   while((element = (TStreamerElement*)next())) {
      if (element->GetOffset() != offset) continue;
      Int_t execid = element->GetExecID();
      if (execid > 0) ref->SetBit(execid << 8);
      return;
   }
}

//______________________________________________________________________________
void TFileIO::WriteRefUniqueID(TBuffer &R__b, TObject *obj)
{
   //if the object is referenced, we must save its address/file_pid
   
   UInt_t uid = obj->GetUniqueID();
   TProcessID *pid = TProcessID::GetProcessWithUID(uid,obj);
   //add uid to the TRefTable if there is one
   TRefTable *table = TRefTable::GetRefTable();
   if(table) table->Add(uid, pid);

   //UShort_t pidf = TProcessID::WriteProcessID(R__b,pid);
   WriteProcessID(R__b,pid);
}

//______________________________________________________________________________
Int_t TFileIO::WriteObject(const TObject *obj, const char *name, Int_t option, Int_t bufsize) const
{
   // Write this object to the current directory.
   // The data structure corresponding to this object is serialized.
   // The corresponding buffer is written to the current directory
   // with an associated key with name "name".
   //
   // Writing an object to a file involves the following steps:
   //
   //  -Creation of a support TKey object in the current directory.
   //   The TKey object creates a TBuffer object.
   //
   //  -The TBuffer object is filled via the class::Streamer function.
   //
   //  -If the file is compressed (default) a second buffer is created to
   //   hold the compressed buffer.
   //
   //  -Reservation of the corresponding space in the file by looking
   //   in the TFree list of free blocks of the file.
   //
   //  -The buffer is written to the file.
   //
   //  Bufsize can be given to force a given buffer size to write this object.
   //  By default, the buffersize will be taken from the average buffer size
   //  of all objects written to the current file so far.
   //
   //  If a name is specified, it will be the name of the key.
   //  If name is not given, the name of the key will be the name as returned
   //  by GetName().
   //
   //  The option can be a combination of:
   //    kSingleKey, kOverwrite or kWriteDelete
   //  Using the kOverwrite option a previous key with the same name is
   //  overwritten. The previous key is deleted before writing the new object.
   //  Using the kWriteDelete option a previous key with the same name is
   //  deleted only after the new object has been written. This option
   //  is safer than kOverwrite but it is slower.
   //  The kSingleKey option is only used by TCollection::Write() to write
   //  a container with a single key instead of each object in the container
   //  with its own key.
   //
   //  An object is read from the file into memory via TKey::Read() or
   //  via TObject::Read().
   //
   //  The function returns the total number of bytes written to the file.
   //  It returns 0 if the object cannot be written.
   // destructor

   if (!gFile) {
      Error("Write","No file open");
      return 0;
   }
   if (bufsize) gFile->SetBufferSize(bufsize);
   TString opt = "";
   if (option & kSingleKey)   opt += "SingleKey";
   if (option & kOverwrite)   opt += "OverWrite";
   if (option & kWriteDelete) opt += "WriteDelete";

   Int_t nbytes = gDirectory->WriteTObject(obj,name,opt.Data());
   if (bufsize) gFile->SetBufferSize(0);
   return nbytes;

}

//______________________________________________________________________________
UShort_t TFileIO::WriteProcessID(TBuffer &R__b, TProcessID *pidd)
{
   // Check if the ProcessID pid is already in the file.
   // if not, add it and return the index  number in the local file list

   UShort_t pidf;
   TFile *file = (TFile*)R__b.GetParent();
   if (!file) return 0;
   TProcessID *pid = pidd;
   if (!pid) pid = TProcessID::GetPID();
   TObjArray *pids = file->GetListOfProcessIDs();
   Int_t npids = file->GetNProcessIDs();
   for (Int_t i=0;i<npids;i++) {
      if (pids->At(i) == pid) {
         pidf = (UShort_t)i;
         R__b <<pidf;
         return pidf;
      }
   }

   TDirectory *dirsav = gDirectory;
   file->cd();
   file->SetBit(TFile::kHasReferences);
   pids->AddAtAndExpand(pid,npids);
   pid->IncrementCount();
   char name[32];
   sprintf(name,"ProcessID%d",npids);
   pid->Write(name);
   file->IncrementProcessIDs();
   if (gDebug > 0) {
      printf("WriteProcessID, name=%s, file=%s\n",name,file->GetName());
   }
   if (dirsav) dirsav->cd();
   pidf = (UShort_t)npids;
   R__b <<pidf;
   return pidf;
}
