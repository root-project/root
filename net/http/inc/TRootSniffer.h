// $Id$
// Author: Sergey Linev   22/12/2013

#ifndef ROOT_TRootSniffer
#define ROOT_TRootSniffer

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TList
#include "TList.h"
#endif

enum {
   mask_Scan        = 0x0001,  ///< normal scan of hierarchy
   mask_Expand      = 0x0002,  ///< expand of specified item - allowed to scan object members
   mask_Search      = 0x0004,  ///< search for specified item (only objects and collections)
   mask_CheckChld   = 0x0008,  ///< check if there childs, very similar to search
   mask_Actions     = 0x000F,  ///< mask for actions, only actions copied to child rec
   mask_ExtraFolder = 0x0010   ///< bit marks folder where all childs can be expanded
};

class TMemFile;
class TBufferFile;
class TDataMember;
class TRootSnifferStore;

class TRootSnifferScanRec {
public:

   TRootSnifferScanRec *parent; //! pointer on parent record
   UInt_t mask;                 //! defines operation kind
   const char *searchpath;      //! current path searched
   Int_t lvl;                   //! current level of hierarchy
   TList fItemsNames;           //! list of created items names, need to avoid duplication

   TRootSnifferStore *store;  //! object to store results
   Bool_t has_more;           //! indicates that potentially there are more items can be found
   TString started_node;      //! name of node stared
   Int_t num_fields;          //! number of fields
   Int_t num_childs;          //! number of childs

   TRootSnifferScanRec();
   virtual ~TRootSnifferScanRec();

   void CloseNode();

   /** return true when fields could be set to the hierarchy item */
   Bool_t CanSetFields()
   {
      return (mask & mask_Scan) && (store != 0);
   }

   /** Starts new node, must be closed at the end */
   void CreateNode(const char *_node_name);

   void BeforeNextChild();

   /** Set item field only when creating is specified */
   void SetField(const char *name, const char *value, Bool_t with_quotes = kTRUE);

   /** Mark item with ROOT class and correspondent streamer info */
   void SetRootClass(TClass *cl);

   /** Returns true when item can be expanded */
   Bool_t CanExpandItem();

   /** Checks if result will be accepted. Used to verify if sniffer should read object from the file */
   Bool_t IsReadyForResult() const;

   /** Set result pointer and return true if result is found */
   Bool_t SetResult(void *obj, TClass *cl, TDataMember *member = 0,  Int_t chlds = -1);

   /** Returns depth of hierarchy */
   Int_t Depth() const;

   /** Returns level till extra folder, marked as mask_ExtraFolder */
   Int_t ExtraFolderLevel();

   /** Method indicates that scanning can be interrupted while result is set */
   Bool_t Done() const;

   /** Construct item name, using object name as basis */
   void MakeItemName(const char *objname, TString& itemname);

   Bool_t GoInside(TRootSnifferScanRec &super, TObject *obj, const char *obj_name = 0);

   ClassDef(TRootSnifferScanRec, 0) // Scan record for objects sniffer
};


class TRootSniffer : public TNamed {

protected:

   TString     fObjectsPath; //! path for registered objects
   TMemFile   *fMemFile;     //! file used to manage streamer infos
   Int_t       fSinfoSize;   //! number of elements in streamer info, used as version
   Bool_t      fReadOnly;    //! indicate if sniffer allowed to change ROOT structures - for instance, read objects from files

   void ScanObjectMemebers(TRootSnifferScanRec &rec, TClass *cl, char *ptr, unsigned long int cloffset);

   void ScanObject(TRootSnifferScanRec &rec, TObject *obj);

   virtual void ScanObjectProperties(TRootSnifferScanRec &rec, TObject* &obj, TClass* &obj_class);

   virtual void ScanObjectChilds(TRootSnifferScanRec &rec, TObject *obj);

   void ScanCollection(TRootSnifferScanRec &rec, TCollection *lst,
                       const char *foldername = 0, Bool_t extra = kFALSE, TCollection* keys_lst = 0);

   /* Method is used to scan ROOT objects.
    * Can be reimplemented to extend scanning */
   virtual void ScanRoot(TRootSnifferScanRec &rec);

   void CreateMemFile();

   TString DecodeUrlOptionValue(const char* value, Bool_t remove_quotes = kTRUE);

public:

   TRootSniffer(const char *name, const char *objpath = "online");
   virtual ~TRootSniffer();

   static Bool_t IsDrawableClass(TClass *cl);

   /** When readonly on (default), sniffer is not allowed to change ROOT structures.
    * For instance, it is not allowed to read new objects from files */
   void  SetReadOnly(Bool_t on = kTRUE) { fReadOnly = on; }

   /** Return readonly mode */
   Bool_t IsReadOnly() const { return fReadOnly; }

   Bool_t RegisterObject(const char *subfolder, TObject *obj);

   Bool_t UnregisterObject(TObject *obj);

   /** Method scans normal objects, registered in ROOT */
   void ScanHierarchy(const char *topname, const char *path, TRootSnifferStore *store);

   TObject *FindTObjectInHierarchy(const char *path);

   virtual void *FindInHierarchy(const char *path, TClass **cl = 0, TDataMember **member = 0, Int_t *chld = 0);

   Bool_t CanDrawItem(const char *path);

   Bool_t CanExploreItem(const char *path);

   Bool_t IsStreamerInfoItem(const char *itemname);

   ULong_t GetStreamerInfoHash();

   ULong_t GetItemHash(const char *itemname);

   Bool_t ProduceJson(const char *path, const char *options, TString &res);

   Bool_t ProduceXml(const char *path, const char *options, TString &res);

   Bool_t ProduceBinary(const char *path, const char *options, void *&ptr, Long_t &length);

   Bool_t ProduceImage(Int_t kind, const char *path, const char *options, void *&ptr, Long_t &length);

   Bool_t ProduceExe(const char *path, const char *options, TString &res, Bool_t astxt = kFALSE);

   Bool_t Produce(const char *path, const char *file, const char *options, void *&ptr, Long_t &length);

   ClassDef(TRootSniffer, 0) // Sniffer of ROOT objects
};

#endif
