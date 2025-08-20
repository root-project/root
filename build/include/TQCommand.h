// @(#)root/base:$Id$
// Author: Valeriy Onuchin   04/27/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQCommand
#define ROOT_TQCommand

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQCommand, TQUndoManager - support for multiple Undo/Redo operations //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TList.h"

#include "TQObject.h"

class TQConnection;

class TQCommand : public TList, public TQObject {

friend class TQUndoManager;

protected:
   TQConnection  *fRedo;      // do/redo action
   TQConnection  *fUndo;      // undo action
   Long_t        *fRedoArgs;  // redo values
   Long_t        *fUndoArgs;  // undo values
   Int_t          fNRargs;    // number of redo arguments
   Int_t          fNUargs;    // number of undo arguments
   Int_t          fState;     // -1 undoing on, 1 redoing on, 0 nothing in progress
   Int_t          fStatus;    // fStatus++ after Redo(), fStatus-- after Undo()
   Bool_t         fNewDelete; // kTRUE if Redo/Undo methods are new/delete
   TString        fName;      // command name. Default is "ClassName::RedoName(args)"
   TString        fTitle;     // command description
   void          *fObject;    // object to which undo/redo actions applied

   virtual void Init(const char *cl, void *object,
                     const char *redo, const char *undo);
   void PrintCollectionHeader(Option_t* option) const override;

private:
   TQCommand &operator=(const TQCommand &); // Not yet implemented.

public:
   TQCommand(const char *cl = nullptr, void *object = nullptr,
             const char *redo = nullptr, const char *undo = nullptr);
   TQCommand(TObject *obj, const char *redo = nullptr, const char *undo = nullptr);
   TQCommand(const TQCommand &com);
   virtual ~TQCommand();

   virtual void   Redo(Option_t *option="");  //*SIGNAL*
   virtual void   Undo(Option_t *option="");  //*SIGNAL*
   virtual void   SetArgs(Int_t nargs, ...);
   virtual void   SetUndoArgs(Int_t nargs, ...);
   virtual void   SetRedoArgs(Int_t nargs, ...);
   virtual Bool_t CanMerge(TQCommand *c) const;
   virtual void   Merge(TQCommand *c);
   virtual Long64_t Merge(TCollection*,TFileMergeInfo*);
   virtual Bool_t CanCompress(TQCommand *c) const;
   virtual void   Compress(TQCommand *c);
   Bool_t         IsEqual(const TObject* obj) const override;
   virtual Bool_t IsSetter() const;
   virtual Bool_t CanRedo() const;
   virtual Bool_t CanUndo() const;
   const char    *GetRedoName() const;
   const char    *GetUndoName() const;
   TQConnection  *GetRedo() const { return fRedo; }
   TQConnection  *GetUndo() const { return fUndo; }
   Long_t        *GetRedoArgs() const;
   Long_t        *GetUndoArgs() const;
   Int_t          GetNRargs() const;
   Int_t          GetNUargs() const;
   void          *GetObject() const;
   Int_t          GetStatus() const;
   Bool_t         IsMacro() const;
   Bool_t         IsUndoing() const;
   Bool_t         IsRedoing() const;
   Bool_t         IsExecuting() const;
   virtual void   SetName(const char *name);
   virtual void   SetTitle(const char *title);
   void           ls(Option_t *option="") const override;
   void           Add(TObject *obj, Option_t *opt) override;
   void           Add(TObject *obj) override { Add(obj, nullptr); }
   void           Delete(Option_t *option="") override;
   const char    *GetName() const override;
   const char    *GetTitle() const override;

   static TQCommand *GetCommand();

   ClassDefOverride(TQCommand,0) // encapsulates the information for undo/redo a single action.
};


//////////////////////////////////////////////////////////////////////////
class TQUndoManager : public TQCommand {

protected:
   TObjLink   *fCursor;  // current position in history stack
   TQCommand  *fCurrent; // the latest executed command
   UInt_t      fLimit;   // maximum number of commands can be located in stack
   TList      *fLogBook; // listing of all actions during execution
   Bool_t      fLogging; // kTRUE if logging is ON

   void PrintCollectionEntry(TObject* entry, Option_t* option, Int_t recurse) const override;

public:
   TQUndoManager();
   virtual ~TQUndoManager();

   void           Add(TObject *obj, Option_t *opt) override;
   void           Add(TObject *obj) override { Add(obj, nullptr); }
   void           Redo(Option_t *option="") override;
   void           Undo(Option_t *option="") override;
   Bool_t         CanRedo() const override;
   Bool_t         CanUndo() const override;
   virtual void   SetLogging(Bool_t on = kTRUE);
   Bool_t         IsLogging() const;
   TQCommand     *GetCurrent() const;
   TQCommand     *GetCursor() const;
   UInt_t         GetLimit() const;
   virtual void   SetLimit(UInt_t limit);
   virtual void   CurrentChanged(TQCommand *c); //*SIGNAL*
   void           ls(Option_t *option="") const override;

   ClassDefOverride(TQUndoManager,0) // recorder of operations for undo and redo
};

#endif
