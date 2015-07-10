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

class TFolder;
class TMemFile;
class TBufferFile;
class TDataMember;
class THttpCallArg;
class TRootSnifferStore;
class TRootSniffer;

class TRootSnifferScanRec {

friend class TRootSniffer;

protected:
   // different bits used to scan hierarchy
   enum {
      kScan         = 0x0001,  ///< normal scan of hierarchy
      kExpand       = 0x0002,  ///< expand of specified item - allowed to scan object members
      kSearch       = 0x0004,  ///< search for specified item (only objects and collections)
      kCheckChilds  = 0x0008,  ///< check if there childs, very similar to search
      kOnlyFields   = 0x0010,  ///< if set, only fields for specified item will be set (but all fields)
      kActions      = 0x001F   ///< mask for actions, only actions copied to child rec
   };


   TRootSnifferScanRec *fParent;      //! pointer on parent record
   UInt_t               fMask;        //! defines operation kind
   const char          *fSearchPath;  //! current path searched
   Int_t                fLevel;       //! current level of hierarchy
   TString              fItemName;    //! name of current item
   TList                fItemsNames;  //! list of created items names, need to avoid duplication
   Int_t                fRestriction; //! restriction 0 - default, 1 - read-only, 2 - full access

   TRootSnifferStore   *fStore;       //! object to store results
   Bool_t               fHasMore;     //! indicates that potentially there are more items can be found
   Bool_t               fNodeStarted; //! indicate if node was started
   Int_t                fNumFields;   //! number of fields
   Int_t                fNumChilds;   //! number of childs

public:

   TRootSnifferScanRec();
   virtual ~TRootSnifferScanRec();

   void CloseNode();

   /** return true when fields could be set to the hierarchy item */
   Bool_t CanSetFields() const
   {
      return (fMask & kScan) && (fStore != 0);
   }

   Bool_t ScanOnlyFields() const
   {
      return (fMask & kOnlyFields) && (fMask & kScan);
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
   Bool_t SetResult(void *obj, TClass *cl, TDataMember *member = 0);

   /** Returns depth of hierarchy */
   Int_t Depth() const;

   /** Method indicates that scanning can be interrupted while result is set */
   Bool_t Done() const;

   /** Construct item name, using object name as basis */
   void MakeItemName(const char *objname, TString &itemname);

   /** Produces full name for the current item */
   void BuildFullName(TString& buf, TRootSnifferScanRec* prnt = 0);

   /** Returns read-only flag for current item */
   Bool_t IsReadOnly(Bool_t dflt = kTRUE);

   Bool_t GoInside(TRootSnifferScanRec &super, TObject *obj, const char *obj_name = 0, TRootSniffer* sniffer = 0);

   ClassDef(TRootSnifferScanRec, 0) // Scan record for objects sniffer
};

//_______________________________________________________________________

class TRootSniffer : public TNamed {
   enum {
      kItemField = BIT(21)  // item property stored as TNamed
   };
protected:
   TString        fObjectsPath;     //! default path for registered objects
   TMemFile      *fMemFile;         //! file used to manage streamer infos
   TList         *fSinfo;           //! last produced streamer info
   Bool_t         fReadOnly;        //! indicate if sniffer allowed to change ROOT structures - for instance, read objects from files
   Bool_t         fScanGlobalDir;   //! when enabled (default), scan gROOT for histograms, canvases, open files
   THttpCallArg  *fCurrentArg;      //! current http arguments (if any)
   Int_t          fCurrentRestrict; //! current restriction for last-found object
   TString        fCurrentAllowedMethods;  //! list of allowed methods, extracted when analyzed object restrictions
   TList          fRestrictions;    //! list of restrictions for different locations
   TString        fAutoLoad;        //! scripts names, which are add as _autoload parameter to h.json request

   void ScanObjectMemebers(TRootSnifferScanRec &rec, TClass *cl, char *ptr, unsigned long int cloffset);

   virtual void ScanObjectProperties(TRootSnifferScanRec &rec, TObject *obj);

   virtual void ScanObjectChilds(TRootSnifferScanRec &rec, TObject *obj);

   void ScanCollection(TRootSnifferScanRec &rec, TCollection *lst,
                       const char *foldername = 0, TCollection *keys_lst = 0);

   /* Method is used to scan ROOT objects.
    * Can be reimplemented to extend scanning */
   virtual void ScanRoot(TRootSnifferScanRec &rec);

   void CreateMemFile();

   TString DecodeUrlOptionValue(const char *value, Bool_t remove_quotes = kTRUE);

   TObject *GetItem(const char *fullname, TFolder *&parent, Bool_t force = kFALSE, Bool_t within_objects = kTRUE);

   TFolder *GetSubFolder(const char *foldername, Bool_t force = kFALSE);

   const char *GetItemField(TFolder *parent, TObject *item, const char *name);

   Bool_t IsItemField(TObject* obj) const;

   Bool_t AccessField(TFolder *parent, TObject *item,
                      const char *name, const char *value, TNamed **only_get = 0);

   Int_t WithCurrentUserName(const char* option);

public:

   TRootSniffer(const char *name, const char *objpath = "Objects");
   virtual ~TRootSniffer();

   static Bool_t IsDrawableClass(TClass *cl);

   void  SetReadOnly(Bool_t on = kTRUE)
   {
      // When readonly on (default), sniffer is not allowed to change ROOT structures
      // For instance, it is not allowed to read new objects from files

      fReadOnly = on;
   }

   Bool_t IsReadOnly() const
   {
      // Returns readonly mode

      return fReadOnly;
   }

   void Restrict(const char* path, const char* options);

   Bool_t HasRestriction(const char* item_name);

   Int_t CheckRestriction(const char* item_name);

   void SetScanGlobalDir(Bool_t on = kTRUE)
   {
      // When enabled (default), sniffer scans gROOT for files, canvases, histograms

      fScanGlobalDir = on;
   }

   void SetAutoLoad(const char* scripts = "");

   const char* GetAutoLoad() const;

   Bool_t IsScanGlobalDir() const { return fScanGlobalDir; }

   Bool_t RegisterObject(const char *subfolder, TObject *obj);

   Bool_t UnregisterObject(TObject *obj);

   Bool_t RegisterCommand(const char *cmdname, const char *method, const char *icon);

   Bool_t CreateItem(const char *fullname, const char *title);

   Bool_t SetItemField(const char *fullname, const char *name, const char *value);

   const char *GetItemField(const char *fullname, const char *name);

   void SetCurrentCallArg(THttpCallArg* arg);

   /** Method scans normal objects, registered in ROOT */
   void ScanHierarchy(const char *topname, const char *path,
                      TRootSnifferStore *store, Bool_t only_fields = kFALSE);

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

   Bool_t ProduceExe(const char *path, const char *options, Int_t reskind, TString *ret_str, void **ret_ptr = 0, Long_t *ret_length = 0);

   Bool_t ExecuteCmd(const char *path, const char *options, TString &res);

   Bool_t ProduceItem(const char *path, const char *options, TString &res, Bool_t asjson = kTRUE);

   Bool_t ProduceMulti(const char *path, const char *options, void *&ptr, Long_t &length, TString &str, Bool_t asjson = kTRUE);

   Bool_t Produce(const char *path, const char *file, const char *options, void *&ptr, Long_t &length, TString &str);

   ClassDef(TRootSniffer, 0) // Sniffer of ROOT objects
};

#endif
