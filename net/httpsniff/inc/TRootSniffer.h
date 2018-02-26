// $Id$
// Author: Sergey Linev   23/02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootSniffer
#define ROOT_TRootSniffer

#include "TRootSnifferBase.h"

class TMemFile;

class TRootSniffer : public TRootSnifferBase {
protected:
   TMemFile *fMemFile{nullptr}; ///<! file used to manage streamer infos
   TList *fSinfo{nullptr};      ///<! last produced streamer info

   virtual void ScanObjectProperties(TRootSnifferScanRec &rec, TObject *obj);

   virtual void ScanKeyProperties(TRootSnifferScanRec &rec, TKey *key, TObject *&obj, TClass *&obj_class);

   virtual void ScanObjectChilds(TRootSnifferScanRec &rec, TObject *obj);

   void CreateMemFile();

   virtual Bool_t CanDrawClass(TClass *);

   virtual Bool_t HasStreamerInfo() const { return kTRUE; }

public:
   TRootSniffer(const char *name, const char *objpath = "Objects");
   virtual ~TRootSniffer();

   virtual Bool_t IsStreamerInfoItem(const char *itemname);

   virtual ULong_t GetStreamerInfoHash();

   virtual ULong_t GetItemHash(const char *itemname);

   virtual void *FindInHierarchy(const char *path, TClass **cl = nullptr, TDataMember **member = nullptr, Int_t *chld = nullptr);

   virtual Bool_t ProduceBinary(const char *path, const char *options, void *&ptr, Long_t &length);

   virtual Bool_t ProduceImage(Int_t kind, const char *path, const char *options, void *&ptr, Long_t &length);

   virtual Bool_t ProduceXml(const char *path, const char *options, TString &res);

   virtual Bool_t ProduceExe(const char *path, const char *options, Int_t reskind, TString *ret_str,
                             void **ret_ptr = nullptr, Long_t *ret_length = nullptr);

   ClassDef(TRootSniffer, 0) // Sniffer of ROOT objects
};

#endif
