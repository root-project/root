// @(#)root/gviz3d:$Id$
// Author: Tomasz Sosnicki   18/09/09

/************************************************************************
* Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#ifndef ROOT_TStructViewerGUI
#define ROOT_TStructViewerGUI

#include <TGFrame.h>
#include <TGLEmbeddedViewer.h>
#include <TGToolTip.h>
#include <TGLabel.h>
#include <TGNumberEntry.h>
#include <TGeoVolume.h>
#include <TExMap.h>

class TGeoMedium;
class TStructViewer;
class TGeoVolume;
class TStructNode;
class TCanvas;
class TGCheckButton;
class TGTextButton;
class TGRadioButton;
class TStructNodeEditor;
class TStructNodeProperty;
class TGLPhysicalShape;
class TString;
class TGTextEntry;

class TStructViewerGUI : public TGMainFrame {

private:
   TStructViewer       *fParent;                // Pointer to Viewer GUI
   TGeoVolume          *fTopVolume;             // Main volume containing all others volumes
   TStructNode         *fNodePtr;               // Root node which represents the main pointer
   UInt_t               fMaxSlices;             // Maximum number of slices used to build a collection node
   UInt_t               fMouseX;                // Position of ToolTip on x-axis
   UInt_t               fMouseY;                // Position of ToolTip on y-axis
   TStructNode         *fSelectedObject;        // Pointer to actual selected object on scene
   TList                fUndoList;              // List with nodes pointers which were top nodes
   TList                fRedoList;              // List with nodes pointers which were top nodes
   TList                fVisibleObjects;        // List with pointer to nodes which are visible
   Float_t              fMaxRatio;              // Maximum ratio used to scale objetcs
   TList               *fColors;                // Pointer to the list with color properties
   static TGeoMedium   *fgMedium;               // Material and medium
   TExMap               fVolumes;               // Map with pointers to Volumes associated with nodes
   static UInt_t        fgCounter;              // Volume counter

   // layout
   TCanvas             *fCanvas;                // Canvas used to store and paint objects
   TGLEmbeddedViewer   *fGLViewer;              // GLViewer in frame
   TGToolTip           *fToolTip;               // ToolTip is showed when user mouse is over the object
   TGCheckButton       *fShowLinksCheckButton;  // Enable/Disable lines between nodes
   TGLabel             *fNodeNameLabel;         // Label with name of node
   TGLabel             *fNodeTypelabel;         // Label with classname
   TGLabel             *fMembersCountLabel;     // Label with number of members in node
   TGLabel             *fAllMembersCountLabel;  // Label with daugthers members
   TGLabel             *fSizeLabel;             // Label with size of node
   TGLabel             *fTotalSizeLabel;        // Label with size of node and daughters nodes
   TGLabel             *fLevelLabel;            // Label with level where the node is placed
   TGTextButton        *fUndoButton;            // Button which can restore last top node
   TGTextButton        *fRedoButton;            // Button which can repeat last node change
   TGRadioButton       *fScaleBySizeButton;     // Sets sorting method to size
   TGRadioButton       *fScaleByMembersButton;  // Sets sorting method to members
   TGTextEntry         *fPointerTextEntry;      // Sets address of pointer
   TGTextEntry         *fPointerTypeTextEntry;  // Sets type of pointer
   TStructNodeEditor   *fEditor;                // Frame with a node editor
   TGNumberEntry       *fBoxHeightEntry;        // Height of boxes
   TGCheckButton       *fAutoRefesh;            // Automatic redraw the scene
   TGNumberEntry       *fLevelDistanceEntry;    // Distance between levels

private:
   void           CalculatePosistion(TStructNode* parent);
   void           CheckMaxObjects(TStructNode* parent);
   void           Divide(TList* list, Float_t x1, Float_t x2, Float_t y1, Float_t y2);
   void           DrawNode(TStructNode* node);
   void           DrawLink(TStructNode* parent);
   void           DrawVolumes(TStructNode* visObj);
   TStructNodeProperty* FindNodeProperty(TStructNode* node);
   void           Scale(TStructNode* parent);
   void           UnCheckMaxObjects();
   void           UpdateLabels( TStructNode* node );

public:
   TStructViewerGUI(TStructViewer* parent, TStructNode* nodePtr, TList* colors, const TGWindow *p = NULL,
      UInt_t w = 800, UInt_t h = 600);
   ~TStructViewerGUI();

   void           AutoRefreshButtonSlot(Bool_t on);
   void           BoxHeightValueSetSlot(Long_t h);
   void           CloseWindow();
   void           ColorSelectedSlot(Pixel_t pixel);
   void           DoubleClickedSlot();
   void           Draw(Option_t* option = "");
   TCanvas       *GetCanvas();
   Int_t          GetColor(TStructNode* node);
   TStructNodeProperty* GetDefaultColor();
   Bool_t         GetLinksVisibility() const;
   TStructNode   *GetNodePtr() const;
   void           GLWidgetProcessedEventSlot(Event_t* event);
   void           LevelDistValueSetSlot(Long_t dist);
   void           MouseOverSlot(TGLPhysicalShape* shape);
   void           RedoButtonSlot();
   void           ResetButtonSlot();
   void           ScaleByChangedSlot();
   void           SetLinksVisibility(Bool_t val);
   void           SetNodePtr(TStructNode* val);
   void           SetPointerButtonSlot();
   void           ShowLinksToggled(Bool_t on);
   void           UndoButtonSlot();
   void           Update(Bool_t resetCamera = false);
   void           UpdateButtonSlot();

   ClassDef(TStructViewerGUI, 0); // A GUI fo 3D struct viewer
};

#endif
