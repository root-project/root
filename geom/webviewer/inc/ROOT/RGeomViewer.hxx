// @(#)root/eve7:$Id$
// Author: Sergey Linev, 13.12.2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RGeomViewer
#define ROOT7_RGeomViewer

#include <ROOT/RWebDisplayArgs.hxx>
#include <ROOT/RGeomData.hxx>

#include <memory>

class TGeoManager;
class TGeoVolume;

namespace ROOT {
namespace Experimental {

class RWebWindow;

class RGeomViewer {

protected:

   TGeoManager *fGeoManager{nullptr};        ///<! geometry to show
   std::string fSelectedVolume;              ///<! name of selected volume
   RGeomDescription fDesc;                   ///<! geometry description, send to the client as first message
   bool fShowHierarchy{true};                ///<! if hierarchy visible by default
   bool fShowColumns{true};                  ///<! show columns in hierarchy
   std::string fTitle;                       ///<! title of geometry viewer

   std::shared_ptr<RWebWindow> fWebWindow;   ///<! web window to show geometry

   void WebWindowCallback(unsigned connid, const std::string &arg);

   std::vector<int> GetStackFromJson(const std::string &json, bool node_ids = false);

   void SendGeometry(unsigned connid);

public:

   RGeomViewer(TGeoManager *mgr = nullptr, const std::string &volname = "");
   virtual ~RGeomViewer();

   void SetTitle(const std::string &title) { fTitle = title; }
   const std::string &GetTitle() const { return fTitle; }

   std::string GetWindowAddr() const;

   void SetGeometry(TGeoManager *mgr, const std::string &volname = "");

   void SelectVolume(const std::string &volname);

   void SetOnlyVolume(TGeoVolume *vol);

   /** Configures maximal number of visible nodes and faces */
   void SetLimits(int nnodes = 5000, int nfaces = 100000)
   {
      fDesc.SetMaxVisNodes(nnodes);
      fDesc.SetMaxVisFaces(nfaces);
   }

   /** Configures maximal visible level */
   void SetVisLevel(int lvl = 3)
   {
      fDesc.SetVisLevel(lvl);
   }

   void SetTopVisible(bool on = true)
   {
      fDesc.SetTopVisible(on);
   }

   /** Configures default hierarchy browser visibility, only has effect before showing web window */
   void SetShowHierarchy(bool on = true) { fShowHierarchy = on; }

   /** Returns default hierarchy browser visibility */
   bool GetShowHierarchy() const { return fShowHierarchy; }

   void SetShowColumns(bool on = true) { fShowColumns = on; }

   bool GetShowColumns() const { return fShowColumns; }

   void SetDrawOptions(const std::string &opt);

   void Show(const RWebDisplayArgs &args = "", bool always_start_new_browser = false);

   void Update();

   void SaveImage(const std::string &fname = "geometry.png", int width = 0, int height = 0);

};

} // namespace Experimental
} // namespace ROOT

#endif
