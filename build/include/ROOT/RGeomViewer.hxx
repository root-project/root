// Author: Sergey Linev, 13.12.2018

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
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

class RWebWindow;
class RGeomHierarchy;

class RGeomViewer {

protected:

   TGeoManager *fGeoManager{nullptr};        ///<! geometry to show
   std::string fSelectedVolume;              ///<! name of selected volume
   RGeomDescription fDesc;                   ///<! geometry description, send to the client as first message
   bool fShowHierarchy{true};                ///<! if hierarchy visible by default
   bool fShowColumns{true};                  ///<! show columns in hierarchy
   std::string fTitle;                       ///<! title of geometry viewer
   bool fInfoActive{false};                  ///<! true when info page active and node info need to be provided

   std::shared_ptr<RWebWindow> fWebWindow;   ///<! web window to show geometry

   std::shared_ptr<RGeomHierarchy> fWebHierarchy; ///<! web handle for hierarchy part

   void WebWindowCallback(unsigned connid, const std::string &arg);

   void WebWindowDisconnect(unsigned connid);

   std::vector<int> GetStackFromJson(const std::string &json, bool node_ids = false);

   void SendGeometry(unsigned connid = 0, bool first_time = false);

   void ProcessSignal(const std::string &);

public:

   RGeomViewer(TGeoManager *mgr = nullptr, const std::string &volname = "");
   virtual ~RGeomViewer();

   void SetTitle(const std::string &title) { fTitle = title; }
   const std::string &GetTitle() const { return fTitle; }

   std::shared_ptr<RWebWindow> GetWindow() { return fWebWindow; }

   std::string GetWindowAddr() const;

   std::string GetWindowUrl(bool remote);

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

   RGeomDescription &Description() { return fDesc; }

   void SaveAsMacro(const std::string &fname);

   void ClearOnClose(const std::shared_ptr<void> &handle);

};

} // namespace ROOT

#endif
