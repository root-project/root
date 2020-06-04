// $Id$
// Author: Sergey Linev   23/02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootSnifferFull
#define ROOT_TRootSnifferFull

#include "TRootSniffer.h"
#include <string>

class TMemFile;

class TRootSnifferFull : public TRootSniffer {
protected:
   TMemFile *fMemFile{nullptr}; ///<! file used to manage streamer infos
   TList *fSinfo{nullptr};      ///<! last produced streamer info

   virtual void ScanObjectProperties(TRootSnifferScanRec &rec, TObject *obj);

   virtual void ScanKeyProperties(TRootSnifferScanRec &rec, TKey *key, TObject *&obj, TClass *&obj_class);

   virtual void ScanObjectChilds(TRootSnifferScanRec &rec, TObject *obj);

   void CreateMemFile();

   virtual Bool_t CanDrawClass(TClass *cl) { return IsDrawableClass(cl); }

   virtual Bool_t HasStreamerInfo() const { return kTRUE; }

   virtual Bool_t ProduceBinary(const std::string &path, const std::string &options, std::string &res);

   virtual Bool_t ProduceImage(Int_t kind, const std::string &path, const std::string &options, std::string &res);

   virtual Bool_t ProduceXml(const std::string &path, const std::string &options, std::string &res);

   virtual Bool_t ProduceExe(const std::string &path, const std::string &options, Int_t reskind, std::string &res);

public:
   TRootSnifferFull(const char *name, const char *objpath = "Objects");
   virtual ~TRootSnifferFull();

   static Bool_t IsDrawableClass(TClass *cl);

   virtual Bool_t IsStreamerInfoItem(const char *itemname);

   virtual ULong_t GetStreamerInfoHash();

   virtual ULong_t GetItemHash(const char *itemname);

   virtual void *
   FindInHierarchy(const char *path, TClass **cl = nullptr, TDataMember **member = nullptr, Int_t *chld = nullptr);

   ClassDef(TRootSnifferFull, 0) // Sniffer for many ROOT classes, including histograms, graphs, pads and tree
};

#endif
