// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveScene_hxx
#define ROOT_TEveScene_hxx

#include "ROOT/TEveElement.hxx"

class TExMap;

namespace ROOT { namespace Experimental
{

// class TEvePad;
class TEveClient;

/******************************************************************************/
// TEveScene
/******************************************************************************/

class TEveScene : public TEveElementList
{
private:
   TEveScene(const TEveScene&);            // Not implemented
   TEveScene& operator=(const TEveScene&); // Not implemented
   
protected:
   Bool_t       fSmartRefresh  = kTRUE;
   Bool_t       fHierarchical  = kFALSE;

   Bool_t       fAcceptingChanges = kFALSE;
   Bool_t       fChanged          = kFALSE;
   Set_t        fChangedElements;
   // For the following two have to rethink how the hierarchy will be handled.
   // If I remove a parent, i have to remove all the children.
   // So this has to be done right on both sides (on eve element and here).
   // I might need a set, so i can easily check if parent is in the removed / added list already.
   // List_t       fAddedElements;
   // List_t       fRemovedElements;

   std::list<TEveClient*> fSubscribers;

public:
   std::string       fOutputJson;
   std::vector<char> fOutputBinary;
   List_t            fElsWithBinaryData;
   Int_t             fTotalBinarySize;

   // void RetransHierarchicallyRecurse(TEveElement* el, const TEveTrans& tp);

public:
   TEveScene(const char* n="TEveScene", const char* t="");
   virtual ~TEveScene();

   virtual void CollectSceneParents(List_t& scenes);

   virtual Bool_t SingleRnrState() const { return kTRUE; }

   void   SetHierarchical(Bool_t h) { fHierarchical = h;    }
   Bool_t GetHierarchical()   const { return fHierarchical; }

   void   Changed()         { fChanged = kTRUE; }
   Bool_t IsChanged() const { return fChanged;  }

   Bool_t IsAcceptingChanges() const { return fAcceptingChanges; }
   void   BeginAcceptingChanges();
   void   SceneElementChanged(TEveElement* element);
   void   EndAcceptingChanges();
   void   ProcessChanges(); // should return net message or talk to gEve about it

   void   StreamElements();
   void   StreamJsonRecurse(TEveElement *el, nlohmann::json &jobj);

   // void   Repaint(Bool_t dropLogicals=kFALSE);
   // void   RetransHierarchically();

   // virtual void Paint(Option_t* option = "");

   // void DestroyElementRenderers(TEveElement* element);
   // void DestroyElementRenderers(TObject* rnrObj);
   void StreamRepresentationChanges();
   void SendChangesToSubscribers();

   Bool_t HasSubscribers() const { return ! fSubscribers.empty(); }
   void   AddSubscriber(TEveClient* sub);
   void   RemoveSubscriber(unsigned int);
   
   ClassDef(TEveScene, 0); // Reve representation of TGLScene.
};


/******************************************************************************/
// TEveSceneList
/******************************************************************************/

class TEveSceneList : public TEveElementList
{
private:
   TEveSceneList(const TEveSceneList&);            // Not implemented
   TEveSceneList& operator=(const TEveSceneList&); // Not implemented

protected:

public:
   TEveSceneList(const char* n="TEveSceneList", const char* t="");
   virtual ~TEveSceneList() {}

   void DestroyScenes();

   // void RepaintChangedScenes(Bool_t dropLogicals);
   // void RepaintAllScenes(Bool_t dropLogicals);

   // void DestroyElementRenderers(TEveElement* element);

   void ProcessSceneChanges();

   ClassDef(TEveSceneList, 0); // List of Scenes providing common operations on TEveScene collections.
};

}}

#endif
