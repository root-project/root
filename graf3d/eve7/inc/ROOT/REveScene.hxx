// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveScene
#define ROOT7_REveScene

#include <ROOT/REveElement.hxx>

#include <vector>
#include <memory>

class TExMap;

namespace ROOT {
namespace Experimental {

class REveClient;

/******************************************************************************/
// REveScene
/******************************************************************************/

class REveScene : public REveElementList {
private:
   REveScene(const REveScene &);            // Not implemented
   REveScene &operator=(const REveScene &); // Not implemented

protected:
   Bool_t fSmartRefresh{kTRUE};
   Bool_t fHierarchical{kFALSE};

   Bool_t fAcceptingChanges{kFALSE};
   Bool_t fChanged{kFALSE};
   Set_t fChangedElements;
   // For the following two have to rethink how the hierarchy will be handled.
   // If I remove a parent, i have to remove all the children.
   // So this has to be done right on both sides (on eve element and here).
   // I might need a set, so i can easily check if parent is in the removed / added list already.
   Set_t fAddedElements;
   std::vector<ElementId_t> fRemovedElements;

   std::vector<std::unique_ptr<REveClient>> fSubscribers;

public:
   std::string fOutputJson;
   std::vector<char> fOutputBinary;
   List_t fElsWithBinaryData;
   Int_t fTotalBinarySize;

   // void RetransHierarchicallyRecurse(REveElement* el, const REveTrans& tp);

public:
   REveScene(const char *n = "REveScene", const char *t = "");
   virtual ~REveScene();

   virtual void CollectSceneParents(List_t &scenes);

   virtual Bool_t SingleRnrState() const { return kTRUE; }

   void SetHierarchical(Bool_t h) { fHierarchical = h; }
   Bool_t GetHierarchical() const { return fHierarchical; }

   void Changed() { fChanged = kTRUE; } // AMT ??? depricated
   Bool_t IsChanged() const;

   Bool_t IsAcceptingChanges() const { return fAcceptingChanges; }
   void BeginAcceptingChanges();
   void SceneElementChanged(REveElement *element);
   void SceneElementAdded(REveElement *element);
   void SceneElementRemoved(ElementId_t id);

   void EndAcceptingChanges();
   void ProcessChanges(); // should return net message or talk to gEve about it

   void StreamElements();
   void StreamJsonRecurse(REveElement *el, nlohmann::json &jobj);

   // void   Repaint(Bool_t dropLogicals=kFALSE);
   // void   RetransHierarchically();

   // virtual void Paint(Option_t* option = "");

   // void DestroyElementRenderers(REveElement* element);
   // void DestroyElementRenderers(TObject* rnrObj);
   void StreamRepresentationChanges();
   void SendChangesToSubscribers();

   Bool_t HasSubscribers() const { return !fSubscribers.empty(); }
   void AddSubscriber(std::unique_ptr<REveClient> &&sub);
   void RemoveSubscriber(unsigned int);

   ClassDef(REveScene, 0); // Reve representation of TGLScene.
};

/******************************************************************************/
// REveSceneList
/******************************************************************************/

class REveSceneList : public REveElementList {
private:
   REveSceneList(const REveSceneList &);            // Not implemented
   REveSceneList &operator=(const REveSceneList &); // Not implemented

protected:
public:
   REveSceneList(const char *n = "REveSceneList", const char *t = "");
   virtual ~REveSceneList() {}

   void DestroyScenes();

   // void RepaintChangedScenes(Bool_t dropLogicals);
   // void RepaintAllScenes(Bool_t dropLogicals);

   // void DestroyElementRenderers(REveElement* element);

   void ProcessSceneChanges();

   ClassDef(REveSceneList, 0); // List of Scenes providing common operations on REveScene collections.
};

} // namespace Experimental
} // namespace ROOT

#endif
