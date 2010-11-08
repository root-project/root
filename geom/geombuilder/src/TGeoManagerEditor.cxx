// @(#):$Id$
// Author: M.Gheata 

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoManagerEditor                                                    //
//
//////////////////////////////////////////////////////////////////////////
/*
   Editor for TGeoManager class. Provides also builder functionality for the 
   main TGeo objects: TGeoVolume, TGeoShape - derived classes, TGeoMaterial,
   TGeoMatrix - derived transformations and TGeoMedium.
   The GUI represents the main entry point for editing geometry classes. It
   can be started either by:
   1. TGeoManager::Edit(). The method must be used when starting from a new 
   geometry.
   2. Left-click on the 40x40 pixels top-right corner of a pad containing a
   drawn volume. The region is always accesible when drawing geometry elements 
   and allows also restoring the manager editor in the "Style" tab of the GED
   editor anytime.
   
   The TGeoManager editor is vertically split by a TGShutter widget into the
   following categories:
   
   - General. This allows changing the name/title of the geometry, setting the
   top volume, closing the geometry and saving the geometry in a file. The name
   of the geometry file is formed by geometry_name.C/.root depending if the geometry
   need to be saved as a C macro or a .root file.
   - Shapes. The category provide buttons for creation of all supported shapes. The 
   new shape name is chosen by the interface, but can be changed from the shape 
   editor GUI. Existing shapes can be browsed and edited from the same category. 
   - Volumes. The category allows the creation of a new volume having a given name,
   shape and medium. For creating a volume assembly only the name is relevant. 
   Existing volumes can be browsed or edited from this category.
   - Materials. Allows creation of new materials/mixtures or editing existing ones.
   - Media. The same for creation/editing of tracking media (materials having a set
   of properties related to tracking)
   - Matrices. Allows creation of translations, rotations or combined transformations.
   Existing matrices can also be browser/edited.   
*/   

#include "TVirtualPad.h"
#include "TCanvas.h"
#include "TBaseClass.h"
#include "TGTab.h"
#include "TG3DLine.h"
#include "TGComboBox.h"
#include "TGButton.h"
#include "TGButtonGroup.h"
#include "TGTextEntry.h"
#include "TGNumberEntry.h"
#include "TGLabel.h"
#include "TGShutter.h"

#include "TGeoVolumeEditor.h"
#include "TGeoNodeEditor.h"
#include "TGeoTabManager.h"
#include "TGeoVolume.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoBBox.h"
#include "TGeoPara.h"
#include "TGeoArb8.h"
#include "TGeoTube.h"
#include "TGeoEltu.h"
#include "TGeoHype.h"
#include "TGeoTorus.h"
#include "TGeoTrd1.h"
#include "TGeoTrd2.h"
#include "TGeoCone.h"
#include "TGeoSphere.h"
#include "TGeoPcon.h"
#include "TGeoPgon.h"
#include "TGeoElement.h"
#include "TGeoMaterial.h"
#include "TView.h"

#include "TGeoManagerEditor.h"
#include "TGedEditor.h"

ClassImp(TGeoManagerEditor)

enum ETGeoVolumeWid {
   kMANAGER_NAME, kMANAGER_TITLE, 
   kMANAGER_SHAPE_SELECT, kMANAGER_MEDIA_SELECT,kMANAGER_MATERIAL_SELECT, kMANAGER_ELEMENT_SELECT,
   kMANAGER_SHAPE_SELECT2, kMANAGER_MEDIUM_SELECT2, kMANAGER_VOLUME_SELECT,
   kMANAGER_EDIT_SHAPE, kMANAGER_EDIT_MEDIUM, kMANAGER_DENSITY_SELECT, kMANAGER_NELEM_SELECT,
   kMANAGER_MATERIAL_SELECT2, kMANAGER_MEDIUM_SELECT, kMANAGER_MATRIX_SELECT, kMANAGER_TOP_SELECT,
   kEXPORT_ROOT, kEXPORT_C, kEXPORT_GEOMETRY,
   kCAT_GENERAL, kCAT_SHAPES, kCAT_VOLUMES, kCAT_MEDIA, kCAT_MATERIALS, kCAT_MATRICES,
   kCREATE_BOX, kCREATE_PARA, kCREATE_TRD1, kCREATE_TRD2,
   kCREATE_TRAP, kCREATE_GTRA, kCREATE_XTRU, kCREATE_ARB8,
   kCREATE_TUBE, kCREATE_TUBS, kCREATE_CONE, kCREATE_CONS,
   kCREATE_SPHE, kCREATE_CTUB, kCREATE_ELTU, kCREATE_TORUS,
   kCREATE_PCON, kCREATE_PGON, kCREATE_HYPE, kCREATE_PARAB, kCREATE_COMP,
   kCREATE_MATERIAL, kCREATE_MIXTURE, kCREATE_MEDIUM, kCREATE_VOLUME, kCREATE_ASSEMBLY,
   kCREATE_TRANSLATION, kCREATE_ROTATION, kCREATE_COMBI,
   kMEDIUM_NAME, kMEDIUM_ID, kMATRIX_NAME, kMATERIAL_NAME, kVOLUME_NAME,
   kMANAGER_APPLY, kMANAGER_CANCEL, kMANAGER_UNDO
};

//______________________________________________________________________________
TGeoManagerEditor::TGeoManagerEditor(const TGWindow *p, Int_t width,
                                     Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   // Constructor for manager editor.
   fGeometry = gGeoManager;
   fTabMgr   = 0;
   fTab      = 0;
   fConnectedCanvas = 0;

   fIsModified = kFALSE;   
   TGCompositeFrame *f1;
   TGLabel *label;
   
   // TGShutter for categories
   fCategories = new TGShutter(this, kSunkenFrame | kFixedHeight);

   TGCompositeFrame *container;
   Pixel_t color;
   // General settings
   TGShutterItem *si = new TGShutterItem(fCategories, new TGHotString("General"),kCAT_GENERAL);
   container = (TGCompositeFrame*)si->GetContainer();
   container->SetBackgroundColor(GetDefaultFrameBackground());
   fCategories->AddItem(si);
   // TextEntry for manager name
   f1 = new TGCompositeFrame(container, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(label = new TGLabel(f1, "Name/Title"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   gClient->GetColorByName("#ff0000", color);
   label->SetTextColor(color);
   container->AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 2, 0));
   fManagerName = new TGTextEntry(container, new TGTextBuffer(50), kMANAGER_NAME);
   fManagerName->Resize(135, fManagerName->GetDefaultHeight());
   fManagerName->SetToolTipText("Enter the geometry name");
   container->AddFrame(fManagerName, new TGLayoutHints(kLHintsLeft, 3, 1, 0, 0));
   fManagerTitle = new TGTextEntry(container, new TGTextBuffer(50), kMANAGER_TITLE);
   fManagerTitle->Resize(135, fManagerTitle->GetDefaultHeight());
   fManagerTitle->SetToolTipText("Enter the geometry name");
   container->AddFrame(fManagerTitle, new TGLayoutHints(kLHintsLeft, 3, 1, 0, 0));
   f1 = new TGCompositeFrame(container, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(label = new TGLabel(f1, "Export geometry"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   gClient->GetColorByName("#ff0000", color);
   label->SetTextColor(color);
   container->AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 6, 0));
   TString stitle = "Options";
   f1 = new TGCompositeFrame(container, 155, 10, kHorizontalFrame | kFixedWidth);
   TGButtonGroup *bg = new TGVButtonGroup(f1, stitle);
   fExportOption[0] = new TGRadioButton(bg, ".root", kEXPORT_ROOT);
   fExportOption[1] = new TGRadioButton(bg, ".C", kEXPORT_C);
   fExportButton = new TGTextButton(f1, "Export", kEXPORT_GEOMETRY);
   bg->SetRadioButtonExclusive();
   bg->SetButton(kEXPORT_ROOT);
   bg->Show();
   f1->AddFrame(bg, new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(fExportButton, new TGLayoutHints(kLHintsLeft, 20, 2, 22, 0));
   container->AddFrame(f1, new TGLayoutHints(kLHintsTop | kLHintsLeft, 2, 2, 2, 2));
   // Close geometry
   f7 = new TGCompositeFrame(container, 155, 10, kVerticalFrame | kFixedWidth);
   f1 = new TGCompositeFrame(f7, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(label = new TGLabel(f1, "Close geometry"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   gClient->GetColorByName("#ff0000", color);
   label->SetTextColor(color);
   f7->AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 2, 0));
   f1 = new TGCompositeFrame(f7, 155, 30, kHorizontalFrame | kFixedWidth);
   fLSelTop = new TGLabel(f1, "Select top");
   gClient->GetColorByName("#0000ff", color);
   fLSelTop->SetTextColor(color);
   fLSelTop->ChangeOptions(kSunkenFrame | kDoubleBorder);
   f1->AddFrame(fLSelTop, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY, 1, 1, 2, 2));
   fBSelTop = new TGPictureButton(f1, fClient->GetPicture("rootdb_t.xpm"), kMANAGER_TOP_SELECT);
   fBSelTop->SetToolTipText("Select the top volume");
   fBSelTop->Associate(this);
   f1->AddFrame(fBSelTop, new TGLayoutHints(kLHintsLeft, 1, 1, 2, 2));
   fCloseGeometry = new TGTextButton(f1, "Close");
   f1->AddFrame(fCloseGeometry, new TGLayoutHints(kLHintsRight, 1, 1, 2, 2));
   fCloseGeometry->SetToolTipText("Close geometry to make it ready for tracking");
   fCloseGeometry->Associate(this);
   f7->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
   container->AddFrame(f7, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
   

   si = new TGShutterItem(fCategories, new TGHotString("Shapes"),kCAT_SHAPES);
   container = (TGCompositeFrame*)si->GetContainer();
   container->SetBackgroundColor(GetDefaultFrameBackground());
   fCategories->AddItem(si);

   // Shape creators
   f1 = new TGCompositeFrame(container, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(label = new TGLabel(f1, "Create new shape"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   gClient->GetColorByName("#ff0000", color);
   label->SetTextColor(color);
   container->AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 2, 0));
   TGLayoutHints *lhb = new TGLayoutHints(kLHintsLeft, 0, 4, 0, 0);
   TGLayoutHints *lhf1 = new TGLayoutHints(kLHintsTop | kLHintsLeft, 2, 2, 2, 2);
   Int_t ipict;
   f1 = new TGCompositeFrame(container, 118, 30, kHorizontalFrame);
   fShapeButton[0] = new TGPictureButton(f1, fClient->GetPicture("geobbox_t.xpm"), kCREATE_BOX);
   fShapeButton[0]->SetToolTipText("Create a box");
   fShapeButton[1] = new TGPictureButton(f1, fClient->GetPicture("geopara_t.xpm"), kCREATE_PARA);
   fShapeButton[1]->SetToolTipText("Create a parallelipiped");
   fShapeButton[2] = new TGPictureButton(f1, fClient->GetPicture("geotrd1_t.xpm"), kCREATE_TRD1);
   fShapeButton[2]->SetToolTipText("Create a TRD1 trapezoid");
   fShapeButton[3] = new TGPictureButton(f1, fClient->GetPicture("geotrd2_t.xpm"), kCREATE_TRD2);
   fShapeButton[3]->SetToolTipText("Create a TRD2 trapezoid");
   fShapeButton[4] = new TGPictureButton(f1, fClient->GetPicture("geotrap_t.xpm"), kCREATE_TRAP);
   fShapeButton[4]->SetToolTipText("Create a general trapezoid");
   fShapeButton[5] = new TGPictureButton(f1, fClient->GetPicture("geogtra_t.xpm"), kCREATE_GTRA);
   fShapeButton[5]->SetToolTipText("Create a general twisted trapezoid");
   for (ipict=0; ipict<6; ipict++) f1->AddFrame(fShapeButton[ipict],lhb);
   container->AddFrame(f1, lhf1);
   f1 = new TGCompositeFrame(container, 118, 30, kHorizontalFrame);
   fShapeButton[6] = new TGPictureButton(f1, fClient->GetPicture("geoxtru_t.xpm"), kCREATE_XTRU);
   fShapeButton[6]->SetToolTipText("Create a extruded polygone");
   fShapeButton[7] = new TGPictureButton(f1, fClient->GetPicture("geoarb8_t.xpm"), kCREATE_ARB8);
   fShapeButton[7]->SetToolTipText("Create an arbitrary trapezoid with 8 vertices");
   fShapeButton[8] = new TGPictureButton(f1, fClient->GetPicture("geotube_t.xpm"), kCREATE_TUBE);
   fShapeButton[8]->SetToolTipText("Create a cylindrical pipe");
   fShapeButton[9] = new TGPictureButton(f1, fClient->GetPicture("geotubeseg_t.xpm"), kCREATE_TUBS);
   fShapeButton[9]->SetToolTipText("Create a cylindrical pipe within a phi range");
   fShapeButton[10] = new TGPictureButton(f1, fClient->GetPicture("geocone_t.xpm"), kCREATE_CONE);
   fShapeButton[10]->SetToolTipText("Create a conical pipe");
   fShapeButton[11] = new TGPictureButton(f1, fClient->GetPicture("geoconeseg_t.xpm"), kCREATE_CONS);
   fShapeButton[11]->SetToolTipText("Create a conical pipe within a phi range");
   for (ipict=0; ipict<6; ipict++) f1->AddFrame(fShapeButton[ipict+6],lhb);
   container->AddFrame(f1, lhf1);
   f1 = new TGCompositeFrame(container, 118, 30, kHorizontalFrame);
   fShapeButton[12] = new TGPictureButton(f1, fClient->GetPicture("geosphere_t.xpm"), kCREATE_SPHE);
   fShapeButton[12]->SetToolTipText("Create a spherical sector");
   fShapeButton[13] = new TGPictureButton(f1, fClient->GetPicture("geoctub_t.xpm"), kCREATE_CTUB);
   fShapeButton[13]->SetToolTipText("Create a cut tube");
   fShapeButton[14] = new TGPictureButton(f1, fClient->GetPicture("geoeltu_t.xpm"), kCREATE_ELTU);
   fShapeButton[14]->SetToolTipText("Create an eliptical tube");
   fShapeButton[15] = new TGPictureButton(f1, fClient->GetPicture("geotorus_t.xpm"), kCREATE_TORUS);
   fShapeButton[15]->SetToolTipText("Create a toroidal tube with a phi range");
   fShapeButton[16] = new TGPictureButton(f1, fClient->GetPicture("geopcon_t.xpm"), kCREATE_PCON);
   fShapeButton[16]->SetToolTipText("Create a polycone shape");
   fShapeButton[17] = new TGPictureButton(f1, fClient->GetPicture("geopgon_t.xpm"), kCREATE_PGON);
   fShapeButton[17]->SetToolTipText("Create a polygon shape");
   for (ipict=0; ipict<6; ipict++) f1->AddFrame(fShapeButton[ipict+12],lhb);
   container->AddFrame(f1, lhf1);
   f1 = new TGCompositeFrame(container, 118, 30, kHorizontalFrame);
   fShapeButton[18] = new TGPictureButton(f1, fClient->GetPicture("geohype_t.xpm"), kCREATE_HYPE);
   fShapeButton[18]->SetToolTipText("Create a hyperboloid");
   fShapeButton[19] = new TGPictureButton(f1, fClient->GetPicture("geoparab_t.xpm"), kCREATE_PARAB);
   fShapeButton[19]->SetToolTipText("Create a paraboloid");
   fShapeButton[20] = new TGPictureButton(f1, fClient->GetPicture("geocomposite_t.xpm"), kCREATE_COMP);
   fShapeButton[20]->SetToolTipText("Create a composite shape");
   for (ipict=0; ipict<3; ipict++) f1->AddFrame(fShapeButton[ipict+18],lhb);
   container->AddFrame(f1, lhf1);

   // List of shapes
   f2 = new TGCompositeFrame(container, 155, 10, kVerticalFrame | kFixedWidth);
   f1 = new TGCompositeFrame(f2, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(label = new TGLabel(f1, "Existing shapes"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   gClient->GetColorByName("#ff0000", color);
   label->SetTextColor(color);
   f2->AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 0, 0));
   f1 = new TGCompositeFrame(f2, 155, 30, kHorizontalFrame | kFixedWidth);
   fSelectedShape = 0;
   fLSelShape = new TGLabel(f1, "Select shape");
   gClient->GetColorByName("#0000ff", color);
   fLSelShape->SetTextColor(color);
   fLSelShape->ChangeOptions(kSunkenFrame | kDoubleBorder);
   f1->AddFrame(fLSelShape, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY, 1, 1, 2, 2));
   fBSelShape = new TGPictureButton(f1, fClient->GetPicture("rootdb_t.xpm"), kMANAGER_SHAPE_SELECT);
   fBSelShape->SetToolTipText("Select one of the existing shapes");
   fBSelShape->Associate(this);
   f1->AddFrame(fBSelShape, new TGLayoutHints(kLHintsLeft, 1, 1, 2, 2));
   fEditShape = new TGTextButton(f1, "Edit");
   f1->AddFrame(fEditShape, new TGLayoutHints(kLHintsRight, 1, 1, 2, 2));
   fEditShape->SetToolTipText("Edit selected shape");
   fEditShape->Associate(this);   
   f2->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
   container->AddFrame(f2, new TGLayoutHints(kLHintsLeft, 0, 0, 6, 0));
   
   // Volumes category
   si = new TGShutterItem(fCategories, new TGHotString("Volumes"),kCAT_VOLUMES);
   container = (TGCompositeFrame*)si->GetContainer();
   container->SetBackgroundColor(GetDefaultFrameBackground());
   fCategories->AddItem(si);

   f1 = new TGCompositeFrame(container, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(label = new TGLabel(f1, "Create new volume"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   gClient->GetColorByName("#ff0000", color);
   label->SetTextColor(color);
   container->AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 2, 0));

   f1 = new TGCompositeFrame(container, 155, 30, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(new TGLabel(f1, "Name"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fVolumeName = new TGTextEntry(f1, new TGTextBuffer(50), kVOLUME_NAME);
   fVolumeName->Resize(100, fVolumeName->GetDefaultHeight());
   fVolumeName->SetToolTipText("Enter the name for the new volume");
   f1->AddFrame(fVolumeName, new TGLayoutHints(kLHintsRight, 3, 1, 2, 5));
   container->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 0, 0));

   // ComboBox for shape component
   f1 = new TGCompositeFrame(container, 155, 30, kHorizontalFrame | kFixedWidth);
   fSelectedShape2 = 0;
   fLSelShape2 = new TGLabel(f1, "Select shape");
   gClient->GetColorByName("#0000ff", color);
   fLSelShape2->SetTextColor(color);
   fLSelShape2->ChangeOptions(kSunkenFrame | kDoubleBorder);
   f1->AddFrame(fLSelShape2, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY, 1, 1, 2, 2));
   fBSelShape2 = new TGPictureButton(f1, fClient->GetPicture("rootdb_t.xpm"), kMANAGER_SHAPE_SELECT2);
   fBSelShape2->SetToolTipText("Select one of the existing shapes");
   fBSelShape2->Associate(this);
   f1->AddFrame(fBSelShape2, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   container->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 0, 0));

   // ComboBox for medium component
   f1 = new TGCompositeFrame(container, 155, 30, kHorizontalFrame | kFixedWidth);
   fSelectedMedium2 = 0;
   fLSelMedium2 = new TGLabel(f1, "Select medium");
   gClient->GetColorByName("#0000ff", color);
   fLSelMedium2->SetTextColor(color);
   fLSelMedium2->ChangeOptions(kSunkenFrame | kDoubleBorder);
   f1->AddFrame(fLSelMedium2, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY, 1, 1, 2, 2));
   fBSelMedium2 = new TGPictureButton(f1, fClient->GetPicture("rootdb_t.xpm"), kMANAGER_MEDIUM_SELECT2);
   fBSelMedium2->SetToolTipText("Select one of the existing media");
   fBSelMedium2->Associate(this);
   f1->AddFrame(fBSelMedium2, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   container->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 0, 0));
   // Picture buttons for different volumes
   f1 = new TGCompositeFrame(container, 155, 30, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(new TGLabel(f1, "Create..."), new TGLayoutHints(kLHintsLeft, 1, 5, 6, 0));
   fVolumeButton[0] = new TGPictureButton(f1, fClient->GetPicture("geovolume_t.xpm"), kCREATE_VOLUME);
   fVolumeButton[0]->SetToolTipText("Create a new volume from shape and medium");
   fVolumeButton[1] = new TGPictureButton(f1, fClient->GetPicture("geoassembly_t.xpm"), kCREATE_ASSEMBLY);
   fVolumeButton[1]->SetToolTipText("Create a new volume assemby having the selected name");
   for (ipict=0; ipict<2; ipict++) f1->AddFrame(fVolumeButton[ipict],lhb);
   container->AddFrame(f1, lhf1);
   // List of volumes
   f3 = new TGCompositeFrame(container, 155, 10, kVerticalFrame | kFixedWidth);
   f1 = new TGCompositeFrame(f3, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(label = new TGLabel(f1, "Existing volumes"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   gClient->GetColorByName("#ff0000", color);
   label->SetTextColor(color);
   f3->AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 0, 0));
   f1 = new TGCompositeFrame(f3, 155, 30, kHorizontalFrame | kFixedWidth);
   fSelectedVolume = 0;
   fLSelVolume = new TGLabel(f1, "Select volume");
   gClient->GetColorByName("#0000ff", color);
   fLSelVolume->SetTextColor(color);
   fLSelVolume->ChangeOptions(kSunkenFrame | kDoubleBorder);
   f1->AddFrame(fLSelVolume, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY, 1, 1, 2, 2));
   fBSelVolume = new TGPictureButton(f1, fClient->GetPicture("rootdb_t.xpm"), kMANAGER_VOLUME_SELECT);
   fBSelVolume->SetToolTipText("Select one of the existing volumes");
   fBSelVolume->Associate(this);
   f1->AddFrame(fBSelVolume, new TGLayoutHints(kLHintsRight, 1, 1, 2, 2));
   f3->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 0, 0));
   f1 = new TGCompositeFrame(f3, 155, 30, kHorizontalFrame | kFixedWidth);
   fEditVolume = new TGTextButton(f1, " Edit ");
   f1->AddFrame(fEditVolume, new TGLayoutHints(kLHintsLeft, 20, 1, 2, 2));
   fEditVolume->SetToolTipText("Edit selected volume");
   fEditVolume->Associate(this);
   fSetTopVolume = new TGTextButton(f1, "Set top");
   f1->AddFrame(fSetTopVolume, new TGLayoutHints(kLHintsRight, 1, 20, 2, 2));
   fSetTopVolume->SetToolTipText("Set top volume for geometry");
   fSetTopVolume->Associate(this);
   f3->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
   container->AddFrame(f3, new TGLayoutHints(kLHintsLeft, 0, 0, 6, 0));

   // Materials category
   si = new TGShutterItem(fCategories, new TGHotString("Materials"),kCAT_MATERIALS);
   container = (TGCompositeFrame*)si->GetContainer();
   container->SetBackgroundColor(GetDefaultFrameBackground());
   fCategories->AddItem(si);

   // Material creators
   gGeoManager->BuildDefaultMaterials();
   f1 = new TGCompositeFrame(container, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(label = new TGLabel(f1, "Create material/mixt."), new TGLayoutHints(kLHintsLeft, 2, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   gClient->GetColorByName("#ff0000", color);
   label->SetTextColor(color);
   container->AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 2, 0));
   f1 = new TGCompositeFrame(container, 155, 30, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(new TGLabel(f1, "Name"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMaterialName = new TGTextEntry(f1, new TGTextBuffer(50), kMATERIAL_NAME);
   fMaterialName->Resize(100, fMaterialName->GetDefaultHeight());
   fMaterialName->SetToolTipText("Enter the new material name");
   f1->AddFrame(fMaterialName, new TGLayoutHints(kLHintsRight, 3, 1, 2, 5));
   container->AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 2, 0));

   f1 = new TGCompositeFrame(container, 155, 30, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(new TGLabel(f1, "Element"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fElementList = new TGComboBox(f1, kMANAGER_ELEMENT_SELECT);
   fElementList->Resize(100, fManagerName->GetDefaultHeight());
   TGeoElementTable *table = gGeoManager->GetElementTable();
   if (table) {
      TGeoElement *element;
      for (Int_t i=0; i<table->GetNelements(); i++) {
         element = table->GetElement(i);
         fElementList->AddEntry(element->GetTitle(),i);
      }
   }      
   fElementList->Select(0);
   f1->AddFrame(fElementList, new TGLayoutHints(kLHintsRight, 2, 2, 4, 4));
   container->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 2, 2));
   // Number entry for density
   f1 = new TGCompositeFrame(container, 155, 30, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(new TGLabel(f1, "Density"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fEntryDensity = new TGNumberEntry(f1, 0., 5, kMANAGER_DENSITY_SELECT);
   fEntryDensity->SetNumStyle(TGNumberFormat::kNESRealThree);
   fEntryDensity->SetNumAttr(TGNumberFormat::kNEANonNegative);
   fEntryDensity->Resize(100,fEntryDensity->GetDefaultHeight()); 
   TGTextEntry *nef = (TGTextEntry*)fEntryDensity->GetNumberEntry();
   nef->SetToolTipText("Enter material/mixture density");
   fEntryDensity->SetNumber(0);
   fEntryDensity->Associate(this);
   f1->AddFrame(fEntryDensity, new TGLayoutHints(kLHintsRight, 2, 2, 2, 2));
   container->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 2, 2));
   // Buttons for creating materials/mixtures
   // Picture buttons for different volumes
   f1 = new TGCompositeFrame(container, 155, 30, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(new TGLabel(f1, "Create..."), new TGLayoutHints(kLHintsLeft, 1, 5, 6, 0));
   fMaterialButton[0] = new TGPictureButton(f1, fClient->GetPicture("geomaterial_t.xpm"), kCREATE_MATERIAL);
   fMaterialButton[0]->SetToolTipText("Create a new material from element and density");
   fMaterialButton[1] = new TGPictureButton(f1, fClient->GetPicture("geomixture_t.xpm"), kCREATE_MIXTURE);
   fMaterialButton[1]->SetToolTipText("Create a new mixture with selected density");
   for (ipict=0; ipict<2; ipict++) f1->AddFrame(fMaterialButton[ipict],lhb);
   container->AddFrame(f1, lhf1);
   
   // List of materials
   f4 = new TGCompositeFrame(container, 155, 10, kVerticalFrame | kFixedWidth);
   f1 = new TGCompositeFrame(f4, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(label = new TGLabel(f1, "Existing materials"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   gClient->GetColorByName("#ff0000", color);
   label->SetTextColor(color);
   f4->AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 0, 0));
   f1 = new TGCompositeFrame(f4, 155, 30, kHorizontalFrame | kFixedWidth);
   fSelectedMaterial = 0;
   fLSelMaterial = new TGLabel(f1, "Select material");
   gClient->GetColorByName("#0000ff", color);
   fLSelMaterial->SetTextColor(color);
   fLSelMaterial->ChangeOptions(kSunkenFrame | kDoubleBorder);
   f1->AddFrame(fLSelMaterial, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY, 1, 1, 2, 2));
   fBSelMaterial = new TGPictureButton(f1, fClient->GetPicture("rootdb_t.xpm"), kMANAGER_MATERIAL_SELECT);
   fBSelMaterial->SetToolTipText("Select one of the existing materials");
   fBSelMaterial->Associate(this);
   f1->AddFrame(fBSelMaterial, new TGLayoutHints(kLHintsLeft, 1, 1, 2, 2));
   fEditMaterial = new TGTextButton(f1, "Edit");
   f1->AddFrame(fEditMaterial, new TGLayoutHints(kLHintsRight, 1, 1, 2, 2));
   fEditMaterial->SetToolTipText("Edit selected material");
   fEditMaterial->Associate(this);
   f4->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
   container->AddFrame(f4, new TGLayoutHints(kLHintsLeft, 0, 0, 6, 0));
   
   si = new TGShutterItem(fCategories, new TGHotString("Media"),kCAT_MEDIA);
   container = (TGCompositeFrame*)si->GetContainer();
   container->SetBackgroundColor(GetDefaultFrameBackground());
   fCategories->AddItem(si);

   // Media category
   f1 = new TGCompositeFrame(container, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(label = new TGLabel(f1, "Create new medium"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   gClient->GetColorByName("#ff0000", color);
   label->SetTextColor(color);
   container->AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 2, 0));

   f1 = new TGCompositeFrame(container, 155, 30, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(new TGLabel(f1, "Name"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMediumName = new TGTextEntry(f1, new TGTextBuffer(50), kMEDIUM_NAME);
   fMediumName->Resize(60, fMediumName->GetDefaultHeight());
   fMediumName->SetToolTipText("Enter the new medium name");
   f1->AddFrame(fMediumName, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));

   fMediumId = new TGNumberEntry(f1, 0., 5, kMEDIUM_ID);
   fMediumId->SetNumStyle(TGNumberFormat::kNESInteger);
   fMediumId->SetNumAttr(TGNumberFormat::kNEAPositive);
   fMediumId->Resize(35,fMediumId->GetDefaultHeight()); 
   nef = (TGTextEntry*)fMediumId->GetNumberEntry();
   nef->SetToolTipText("Enter medium ID");
   fMediumId->SetNumber(fGeometry->GetListOfMedia()->GetSize());
   fMediumId->Associate(this);
   f1->AddFrame(fMediumId, new TGLayoutHints(kLHintsRight, 2, 2, 2 ,2));
   f1->AddFrame(new TGLabel(f1, "ID"), new TGLayoutHints(kLHintsRight, 1, 1, 6, 0));   
   container->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 4, 0));
   // ComboBox for materials
   f1 = new TGCompositeFrame(container, 155, 30, kHorizontalFrame | kFixedWidth);
   fSelectedMaterial2 = 0;
   fLSelMaterial2 = new TGLabel(f1, "Select material");
   gClient->GetColorByName("#0000ff", color);
   fLSelMaterial2->SetTextColor(color);
   fLSelMaterial2->ChangeOptions(kSunkenFrame | kDoubleBorder);
   f1->AddFrame(fLSelMaterial2, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY, 1, 1, 2, 2));
   fBSelMaterial2 = new TGPictureButton(f1, fClient->GetPicture("rootdb_t.xpm"), kMANAGER_MATERIAL_SELECT2);
   fBSelMaterial2->SetToolTipText("Select one of the existing materials");
   fBSelMaterial2->Associate(this);
   f1->AddFrame(fBSelMaterial2, new TGLayoutHints(kLHintsLeft, 1, 1, 2, 2));
   container->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 0, 0));
   f1 = new TGCompositeFrame(container, 155, 30, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(new TGLabel(f1, "Create..."), new TGLayoutHints(kLHintsLeft, 1, 5, 6, 0));
   fMediumButton = new TGPictureButton(f1, fClient->GetPicture("geomedium_t.xpm"), kCREATE_MEDIUM);
   fMediumButton->SetToolTipText("Create a new medium from selected material");
   fMediumButton->Associate(this);
   f1->AddFrame(fMediumButton, new TGLayoutHints(kLHintsLeft, 5, 2, 2, 2));
   container->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 2, 2));

   // List of media
   f5 = new TGCompositeFrame(container, 155, 10, kVerticalFrame | kFixedWidth);
   f1 = new TGCompositeFrame(f5, 145, 10, kHorizontalFrame | kLHintsExpandX | kFixedWidth | kOwnBackground);
   f1->AddFrame(label = new TGLabel(f1, "Existing media"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   gClient->GetColorByName("#ff0000", color);
   label->SetTextColor(color);
   f5->AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 0, 0));
   f1 = new TGCompositeFrame(f5, 155, 30, kHorizontalFrame | kFixedWidth);
   fSelectedMedium = 0;
   fLSelMedium = new TGLabel(f1, "Select medium");
   gClient->GetColorByName("#0000ff", color);
   fLSelMedium->SetTextColor(color);
   fLSelMedium->ChangeOptions(kSunkenFrame | kDoubleBorder);
   f1->AddFrame(fLSelMedium, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY, 1, 1, 2, 2));
   fBSelMedium = new TGPictureButton(f1, fClient->GetPicture("rootdb_t.xpm"), kMANAGER_MEDIA_SELECT);
   fBSelMedium->SetToolTipText("Select one of the existing media");
   fBSelMedium->Associate(this);
   f1->AddFrame(fBSelMedium, new TGLayoutHints(kLHintsLeft, 1, 1, 2, 2));
   fEditMedium = new TGTextButton(f1, "Edit");
   f1->AddFrame(fEditMedium, new TGLayoutHints(kLHintsRight, 1, 1, 2, 2));
   fEditMedium->SetToolTipText("Edit selected medium");
   fEditMedium->Associate(this);
   f5->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
   container->AddFrame(f5, new TGLayoutHints(kLHintsLeft, 0, 0, 6, 0));

   // Matrix category
   si = new TGShutterItem(fCategories, new TGHotString("Matrices"),kCAT_MATRICES);
   container = (TGCompositeFrame*)si->GetContainer();
   container->SetBackgroundColor(GetDefaultFrameBackground());
   fCategories->AddItem(si);
   // Name entry
   f1 = new TGCompositeFrame(container, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(label = new TGLabel(f1, "Create new matrix"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   gClient->GetColorByName("#ff0000", color);
   label->SetTextColor(color);
   container->AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 2, 0));

   f1 = new TGCompositeFrame(container, 155, 30, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(new TGLabel(f1, "Name"), new TGLayoutHints(kLHintsLeft, 1, 1, 6, 0));
   fMatrixName = new TGTextEntry(f1, new TGTextBuffer(50), kMATRIX_NAME);
   fMatrixName->Resize(100, fMatrixName->GetDefaultHeight());
   fMatrixName->SetToolTipText("Enter the new matrix name");
   f1->AddFrame(fMatrixName, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));
   container->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 2, 2, 4, 0));
   // Picture buttons for different matrices
   f1 = new TGCompositeFrame(container, 155, 30, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(new TGLabel(f1, "Create..."), new TGLayoutHints(kLHintsLeft, 1, 5, 6, 0));
   fMatrixButton[0] = new TGPictureButton(f1, fClient->GetPicture("geotranslation_t.xpm"), kCREATE_TRANSLATION);
   fMatrixButton[0]->SetToolTipText("Create a translation");
   fMatrixButton[1] = new TGPictureButton(f1, fClient->GetPicture("georotation_t.xpm"), kCREATE_ROTATION);
   fMatrixButton[1]->SetToolTipText("Create a rotation");
   fMatrixButton[2] = new TGPictureButton(f1, fClient->GetPicture("geocombi_t.xpm"), kCREATE_COMBI);
   fMatrixButton[2]->SetToolTipText("Create a rotation + translation");
   for (ipict=0; ipict<3; ipict++) f1->AddFrame(fMatrixButton[ipict],lhb);
   container->AddFrame(f1, lhf1);
   // List of matrices
   f6 = new TGCompositeFrame(container, 155, 10, kVerticalFrame | kFixedWidth);
   f1 = new TGCompositeFrame(f6, 155, 10, kHorizontalFrame | kFixedWidth);
   f1->AddFrame(label = new TGLabel(f1, "Existing matrices"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f1->AddFrame(new TGHorizontal3DLine(f1), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   gClient->GetColorByName("#ff0000", color);
   label->SetTextColor(color);
   f6->AddFrame(f1, new TGLayoutHints(kLHintsTop, 0, 0, 2, 0));
   f1 = new TGCompositeFrame(f6, 155, 30, kHorizontalFrame | kFixedWidth);
   fSelectedMatrix = 0;
   fLSelMatrix = new TGLabel(f1, "Select matrix");
   gClient->GetColorByName("#0000ff", color);
   fLSelMatrix->SetTextColor(color);
   fLSelMatrix->ChangeOptions(kSunkenFrame | kDoubleBorder);
   f1->AddFrame(fLSelMatrix, new TGLayoutHints(kLHintsLeft | kLHintsExpandX | kLHintsExpandY, 1, 1, 2, 2));
   fBSelMatrix = new TGPictureButton(f1, fClient->GetPicture("rootdb_t.xpm"), kMANAGER_MATRIX_SELECT);
   fBSelMatrix->SetToolTipText("Select one of the existing matrices");
   fBSelMatrix->Associate(this);
   f1->AddFrame(fBSelMatrix, new TGLayoutHints(kLHintsLeft, 1, 1, 2, 2));
   fEditMatrix = new TGTextButton(f1, "Edit");
   f1->AddFrame(fEditMatrix, new TGLayoutHints(kLHintsRight, 1, 1, 2, 2));
   fEditMatrix->SetToolTipText("Edit selected matrix");
   fEditMatrix->Associate(this);
   f6->AddFrame(f1, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
   container->AddFrame(f6, new TGLayoutHints(kLHintsLeft, 0, 0, 6, 0));
   
   fCategories->Resize(163,370);
   AddFrame(fCategories, new TGLayoutHints(kLHintsLeft, 0, 0, 4, 4));

   fVolumeTab = CreateEditorTabSubFrame("Volume");

   // Set the fTab and dissconnect editor from the canvas.
   fTab = fGedEditor->GetTab();
   TCanvas* edCanvas = fGedEditor->GetCanvas();
   fGedEditor->DisconnectFromCanvas();
   if (edCanvas != fConnectedCanvas) {
      DisconnectSelected();
      if (edCanvas)
         ConnectSelected(edCanvas);
      fConnectedCanvas = edCanvas;
   }
}

//______________________________________________________________________________
TGeoManagerEditor::~TGeoManagerEditor()
{
// Destructor.
   TGCompositeFrame *cont;
   cont = (TGCompositeFrame*)fCategories->GetItem("General")->GetContainer();
   TGeoTabManager::Cleanup(cont);
   fCategories->GetItem("General")->SetCleanup(0);
   cont = (TGCompositeFrame*)fCategories->GetItem("Shapes")->GetContainer();
   TGeoTabManager::Cleanup(cont);
   fCategories->GetItem("Shapes")->SetCleanup(0);
   cont = (TGCompositeFrame*)fCategories->GetItem("Volumes")->GetContainer();
   TGeoTabManager::Cleanup(cont);
   fCategories->GetItem("Volumes")->SetCleanup(0);
   cont = (TGCompositeFrame*)fCategories->GetItem("Materials")->GetContainer();
   TGeoTabManager::Cleanup(cont);
   fCategories->GetItem("Materials")->SetCleanup(0);
   cont = (TGCompositeFrame*)fCategories->GetItem("Media")->GetContainer();
   TGeoTabManager::Cleanup(cont);
   fCategories->GetItem("Media")->SetCleanup(0);
   cont = (TGCompositeFrame*)fCategories->GetItem("Matrices")->GetContainer();
   TGeoTabManager::Cleanup(cont);
   fCategories->GetItem("Matrices")->SetCleanup(0);

   delete fExportOption[0]; delete fExportOption[1];

   Cleanup();   

   if (fTabMgr) {
      fTabMgr->GetVolumeTab()->Cleanup();
      delete fTabMgr;
   }   
}

//______________________________________________________________________________
void TGeoManagerEditor::SelectedSlot(TVirtualPad* /*pad*/, TObject* obj, Int_t event)
{
   // Connected to TCanvas::Selected. TGeoManagerEditor takes this
   // function from TGedEditor and only uses it if obj is a TGeoVolume.

   if (event == kButton1 && obj->InheritsFrom(TGeoVolume::Class())) {
      TGeoVolume* v = (TGeoVolume*) obj;
      fTabMgr->SetVolTabEnabled();
      fTabMgr->SetTab();
      fTabMgr->GetVolumeEditor(v);
      v->Draw();
   }
}

void TGeoManagerEditor::ConnectSelected(TCanvas *c)
{
   // Connect to TCanvas::Selected.

   c->Connect("Selected(TVirtualPad*,TObject*,Int_t)", "TGeoManagerEditor",
              this, "SelectedSlot(TVirtualPad*,TObject*,Int_t)");
}

void TGeoManagerEditor::DisconnectSelected()
{
   // Disconnect from TCanvas::Selected.

   if (fConnectedCanvas)
      Disconnect(fConnectedCanvas, "Selected(TVirtualPad*,TObject*,Int_t)",
                 this, "SelectedSlot(TVirtualPad*,TObject*,Int_t)");

}

//______________________________________________________________________________
void TGeoManagerEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.
   fManagerName->Connect("TextChanged(const char *)", "TGeoManagerEditor", this, "DoName()");
   fManagerTitle->Connect("TextChanged(const char *)", "TGeoManagerEditor", this, "DoName()");
   fExportButton->Connect("Clicked()", "TGeoManagerEditor", this, "DoExportGeometry()");
   fCloseGeometry->Connect("Clicked()", "TGeoManagerEditor", this, "DoCloseGeometry()");
   fShapeButton[0]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateBox()");
   fShapeButton[1]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreatePara()");
   fShapeButton[2]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateTrd1()");
   fShapeButton[3]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateTrd2()");
   fShapeButton[4]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateTrap()");
   fShapeButton[5]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateGtra()");
   fShapeButton[6]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateXtru()");
   fShapeButton[7]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateArb8()");
   fShapeButton[8]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateTube()");
   fShapeButton[9]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateTubs()");
   fShapeButton[10]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateCone()");
   fShapeButton[11]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateCons()");
   fShapeButton[12]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateSphe()");
   fShapeButton[13]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateCtub()");
   fShapeButton[14]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateEltu()");
   fShapeButton[15]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateTorus()");
   fShapeButton[16]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreatePcon()");
   fShapeButton[17]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreatePgon()");
   fShapeButton[18]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateHype()");
   fShapeButton[19]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateParab()");
   fShapeButton[20]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateComposite()");
   fMatrixButton[0]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateTranslation()");
   fMatrixButton[1]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateRotation()");
   fMatrixButton[2]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateCombi()");
   fVolumeButton[0]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateVolume()");
   fVolumeButton[1]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateAssembly()");
   fBSelTop->Connect("Clicked()", "TGeoManagerEditor", this, "DoSelectTopVolume()");
   fBSelVolume->Connect("Clicked()", "TGeoManagerEditor", this, "DoSelectVolume()");
   fBSelShape->Connect("Clicked()", "TGeoManagerEditor", this, "DoSelectShape()");
   fBSelShape2->Connect("Clicked()", "TGeoManagerEditor", this, "DoSelectShape2()");
   fBSelMatrix->Connect("Clicked()", "TGeoManagerEditor", this, "DoSelectMatrix()");
   fBSelMaterial->Connect("Clicked()", "TGeoManagerEditor", this, "DoSelectMaterial()");
   fBSelMaterial2->Connect("Clicked()", "TGeoManagerEditor", this, "DoSelectMaterial2()");
   fBSelMedium->Connect("Clicked()", "TGeoManagerEditor", this, "DoSelectMedium()");
   fBSelMedium2->Connect("Clicked()", "TGeoManagerEditor", this, "DoSelectMedium2()");
   fSetTopVolume->Connect("Clicked()", "TGeoManagerEditor", this, "DoSetTopVolume()");
   fEditShape->Connect("Clicked()", "TGeoManagerEditor", this, "DoEditShape()");
   fEditMedium->Connect("Clicked()", "TGeoManagerEditor", this, "DoEditMedium()");
   fEditMaterial->Connect("Clicked()", "TGeoManagerEditor", this, "DoEditMaterial()");
   fEditMatrix->Connect("Clicked()", "TGeoManagerEditor", this, "DoEditMatrix()");
   fEditVolume->Connect("Clicked()", "TGeoManagerEditor", this, "DoEditVolume()");
   
   fMaterialButton[0]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateMaterial()");
   fMaterialButton[1]->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateMixture()");
   fMediumButton->Connect("Clicked()", "TGeoManagerEditor", this, "DoCreateMedium()");
}

//______________________________________________________________________________
void TGeoManagerEditor::SetModel(TObject* obj)
{
   // Refresh editor according the selected obj.
   fGeometry = (TGeoManager*)obj;
   fManagerName->SetText(fGeometry->GetName());
   fManagerTitle->SetText(fGeometry->GetTitle());
   fMatrixName->SetText(TString::Format("matrix%i", fGeometry->GetListOfMatrices()->GetEntries()));
   fMaterialName->SetText(TString::Format("material%i", fGeometry->GetListOfMaterials()->GetSize()));
   fMediumName->SetText(TString::Format("medium%i", fGeometry->GetListOfMedia()->GetSize()));
   fVolumeName->SetText(TString::Format("volume%i", fGeometry->GetListOfVolumes()->GetEntries()));
   // Check if master volume can be set
   if (fGeometry->GetMasterVolume()) fSetTopVolume->SetEnabled(kFALSE);
   else fSetTopVolume->SetEnabled(kTRUE);
   // Check if geometry is already closed
   if (!fGeometry->IsClosed()) fCloseGeometry->SetEnabled(kTRUE);
   else {
      fCloseGeometry->SetEnabled(kFALSE);
      fBSelTop->SetEnabled(kFALSE);
   }   
   // Check if volumes category can be activated
   if (!fGeometry->GetListOfShapes()->GetEntries() || !fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kFALSE);
   else    
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   if (!fGeometry->GetListOfShapes()->GetEntries()) ShowSelectShape(kFALSE);
   else ShowSelectShape();
   if (!fGeometry->GetListOfVolumes()->GetEntries()) ShowSelectVolume(kFALSE);
   else ShowSelectVolume();
   if (!fGeometry->GetListOfMedia()->GetSize()) ShowSelectMedium(kFALSE);
   else ShowSelectMedium();
   if (!fGeometry->GetListOfMatrices()->GetEntries()) ShowSelectMatrix(kFALSE);
   else ShowSelectMatrix();

   // Check if media category can be activated
   if (!fGeometry->GetListOfMaterials()->GetSize()) {
      fCategories->GetItem("Media")->GetButton()->SetEnabled(kFALSE);
      ShowSelectMaterial(kFALSE);
   } else {
      fCategories->GetItem("Media")->GetButton()->SetEnabled(kTRUE);
      ShowSelectMaterial();
   }   
   
   fTab->SetTab(0);
   fCategories->Layout();
   if (fTabMgr == 0) {
      fTabMgr = TGeoTabManager::GetMakeTabManager(fGedEditor);
      fTabMgr->fVolumeTab  = fVolumeTab;
   }
   if (fInit) ConnectSignals2Slots();
   // SetActive();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoName()
{
// Change name/title of the geometry
   fGeometry->SetName(fManagerName->GetText());
   fGeometry->SetTitle(fManagerTitle->GetText());
}

//______________________________________________________________________________
void TGeoManagerEditor::DoExportGeometry()
{
// Export geometry as .root or .C file
   Bool_t asroot = fExportOption[0]->IsDown();
   TString s = fGeometry->GetName();
   s = s.Strip();
   s.Remove(20);
   const char *name;
   if (asroot) name = TString::Format("%s.root", s.Data());
   else        name = TString::Format("%s.C", s.Data());
   fGeometry->Export(name);
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateBox()
{
// Create a box.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoBBox(TString::Format("box_%i",id), 1., 1., 1.);
   ShowSelectShape();
   // Check if volumes category can be activated
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreatePara()
{
// Create a parallelipiped.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoPara(TString::Format("para_%i",id), 1., 1., 1., 30., 20., 45.);
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateTrd1()
{
// Create a Trd1.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoTrd1(TString::Format("trd1_%i",id), 0.5, 1., 1., 1.);
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateTrd2()
{
// Create a Trd2.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoTrd2(TString::Format("trd2_%i",id), 0.5, 1., 0.5, 1., 1.);
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateTrap()
{
// Create a general trapezoid.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoTrap(TString::Format("trap_%i",id), 1., 15., 45., 0.5, 0.3, 0.5, 30., 0.5, 0.3, 0.5, 30.);
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateGtra()
{
// Create a twisted trapezoid.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoGtra(TString::Format("gtra_%i",id), 1., 15., 45., 45.,0.5, 0.3, 0.5, 30., 0.5, 0.3, 0.5, 30.);
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateXtru()
{
// Create an extruded polygone.
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateArb8()
{
// Create an arbitrary polygone with maximum 8 vertices sitting on 2 parallel
// planes
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateTube()
{
// Create a tube.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoTube(TString::Format("tube_%i",id), 0.5, 1., 1.);
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateTubs()
{
// Create a tube segment.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoTubeSeg(TString::Format("tubs_%i",id), 0.5, 1., 1.,0.,45.);
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateCone()
{
// Create a cone.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoCone(TString::Format("cone_%i",id), 0.5, 0.5, 1., 1.5, 2.);
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateCons()
{
// Create a cone segment.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoConeSeg(TString::Format("cons_%i",id), 0.5, 0.5, 1., 1.5, 2.,0.,45.);
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateSphe()
{
// Create a sphere.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoSphere(TString::Format("sphere_%i",id), 0.5, 1., 0., 180., 0.,360.);
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();   
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateCtub()
{
// Create a cut tube.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoCtub(TString::Format("ctub_%i",id), 0.5, 1., 1.,0.,45.,0.,0.,-1,0.,0.,1);
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateEltu()
{
// Create an eliptical tube.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoEltu(TString::Format("para_%i",id), 1., 2., 1.5 );
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateTorus()
{
// Create a torus shape.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoTorus(TString::Format("torus_%i",id), 10., 1., 1.5, 0, 360.);
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();

}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreatePcon()
{
// Create a polycone shape.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoPcon(TString::Format("pcon_%i",id), 0., 360., 2);
   ((TGeoPcon*)fSelectedShape)->DefineSection(0, -1, 0.5, 1.);
   ((TGeoPcon*)fSelectedShape)->DefineSection(1, 1, 0.2, 0.5);
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();   
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreatePgon()
{
// Create a polygone shape.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoPgon(TString::Format("pgon_%i",id), 0., 360.,6,2);
   ((TGeoPcon*)fSelectedShape)->DefineSection(0, -1, 0.5, 1.);
   ((TGeoPcon*)fSelectedShape)->DefineSection(1, 1, 0.2, 0.5);
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();   
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateHype()
{
// Create a hyperboloid.
   Int_t id = gGeoManager->GetListOfShapes()->GetEntries();
   fSelectedShape = new TGeoHype(TString::Format("hype_%i",id), 1., 15., 2., 30., 5.);
   ShowSelectShape();
   if (fGeometry->GetListOfMedia()->GetSize())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditShape();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateParab()
{
// Create a paraboloid.
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateComposite()
{
// Create a composite shape.
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateMaterial()
{
// Create a new material.
   TGeoElement *el = fGeometry->GetElementTable()->GetElement(fElementList->GetSelected());
   Double_t density = fEntryDensity->GetNumber();
   const char *name = fMaterialName->GetText();
   fSelectedMaterial = new TGeoMaterial(name, el, density);
   ShowSelectMaterial();
   fCategories->GetItem("Media")->GetButton()->SetEnabled(kTRUE);
   DoEditMaterial();
   fMaterialName->SetText(TString::Format("material%i", fGeometry->GetListOfMaterials()->GetSize()));
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateMixture()
{
// Create a new mixture.
   Double_t density = fEntryDensity->GetNumber();
   const char *name = fMaterialName->GetText();
   fSelectedMaterial = new TGeoMixture(name, 1, density);
   ShowSelectMaterial();
   fCategories->GetItem("Media")->GetButton()->SetEnabled(kTRUE);
   DoEditMaterial();
   fMaterialName->SetText(TString::Format("material%i", fGeometry->GetListOfMaterials()->GetSize()));
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateMedium()
{
// Create a new medium.
   Int_t id = fMediumId->GetIntNumber();
   if (!fSelectedMaterial2) return;
   const char *name = fMediumName->GetText();
   fSelectedMedium = new TGeoMedium(name, id, fSelectedMaterial2);
   ShowSelectMedium();
   if (fGeometry->GetListOfShapes()->GetEntries())
      fCategories->GetItem("Volumes")->GetButton()->SetEnabled(kTRUE);
   DoEditMedium();
   fMediumName->SetText(TString::Format("medium%i", fGeometry->GetListOfMedia()->GetSize()));
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateTranslation()
{
// Create a new translation.
   const char *name = fMatrixName->GetText();
   fSelectedMatrix = new TGeoTranslation(name, 0., 0., 0.);
   fSelectedMatrix->SetBit(TGeoMatrix::kGeoTranslation);
   fSelectedMatrix->RegisterYourself();
   ShowSelectMatrix();
   DoEditMatrix();
   fMatrixName->SetText(TString::Format("matrix%i", fGeometry->GetListOfMatrices()->GetEntries()));
}   

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateRotation()
{
// Create a new rotation.
   const char *name = fMatrixName->GetText();
   fSelectedMatrix = new TGeoRotation(name);
   fSelectedMatrix->SetBit(TGeoMatrix::kGeoRotation);
   fSelectedMatrix->RegisterYourself();
   ShowSelectMatrix();
   DoEditMatrix();
   fMatrixName->SetText(TString::Format("matrix%i", fGeometry->GetListOfMatrices()->GetEntries()));
}   

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateVolume()
{
// Create a new volume.
   const char *name = fVolumeName->GetText();
   if (!fSelectedShape2 || !fSelectedMedium2) return;
   fSelectedVolume = new TGeoVolume(name, fSelectedShape2, fSelectedMedium2);
   fLSelVolume->SetText(name);
   ShowSelectVolume();
   DoEditVolume();
   fVolumeName->SetText(TString::Format("volume%i", fGeometry->GetListOfVolumes()->GetEntries()));
}   

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateAssembly()
{
// Create a new volume assembly.
   const char *name = fVolumeName->GetText();
   fSelectedVolume = new TGeoVolumeAssembly(name);
   fLSelVolume->SetText(name);
   ShowSelectVolume();
   DoEditVolume();
   fVolumeName->SetText(TString::Format("volume%i", fGeometry->GetListOfVolumes()->GetEntries()));
}   

//______________________________________________________________________________
void TGeoManagerEditor::DoCreateCombi()
{
// Create a new translation + rotation.
   const char *name = fMatrixName->GetText();
   fSelectedMatrix = new TGeoCombiTrans(name, 0., 0., 0., new TGeoRotation());
   fSelectedMatrix->RegisterYourself();
   fSelectedMatrix->SetBit(TGeoMatrix::kGeoTranslation);
   fSelectedMatrix->SetBit(TGeoMatrix::kGeoRotation);
   ShowSelectMatrix();
   DoEditMatrix();
   fMatrixName->SetText(TString::Format("matrix%i", fGeometry->GetListOfMatrices()->GetEntries()));
}   

//______________________________________________________________________________
void TGeoManagerEditor::DoSetTopVolume()
{
// Set top volume for the geometry.
   if (!fSelectedVolume) return;
   fGeometry->SetTopVolume(fSelectedVolume);
   fSetTopVolume->SetEnabled(kFALSE);
}      

//______________________________________________________________________________
void TGeoManagerEditor::DoEditShape()
{
// Slot for editing selected shape.
   if (!fSelectedShape) return;
   fTabMgr->GetShapeEditor(fSelectedShape);
   fSelectedShape->Draw();
   fTabMgr->GetPad()->GetView()->ShowAxis();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoEditVolume()
{
// Slot for editing selected volume.
   if (!fSelectedVolume) {
      fTabMgr->SetVolTabEnabled(kFALSE);
      return;
   }   
   fTabMgr->SetVolTabEnabled();
   fTabMgr->SetTab();
   fTabMgr->GetVolumeEditor(fSelectedVolume);
   fSelectedVolume->Draw();
}

//______________________________________________________________________________
void TGeoManagerEditor::DoEditMedium()
{
// Slot for editing selected medium.
   if (!fSelectedMedium) return;
   fTabMgr->GetMediumEditor(fSelectedMedium);
}

//______________________________________________________________________________
void TGeoManagerEditor::DoEditMaterial()
{
// Slot for editing selected material.
   if (!fSelectedMaterial) return;
   fTabMgr->GetMaterialEditor(fSelectedMaterial);
} 

//______________________________________________________________________________
void TGeoManagerEditor::DoEditMatrix()
{
// Slot for editing selected matrix.
   if (!fSelectedMatrix) return;
   fTabMgr->GetMatrixEditor(fSelectedMatrix);
}

//______________________________________________________________________________
void TGeoManagerEditor::DoSelectMatrix()
{
// Slot for selecting an existing matrix.
   TGeoMatrix *matrix = fSelectedMatrix;
   new TGeoMatrixDialog(fBSelMatrix, gClient->GetRoot(), 200,300);  
   fSelectedMatrix = (TGeoMatrix*)TGeoMatrixDialog::GetSelected();
   if (fSelectedMatrix) fLSelMatrix->SetText(fSelectedMatrix->GetName());
   else fSelectedMatrix = matrix;
}

//______________________________________________________________________________
void TGeoManagerEditor::DoSelectShape()
{
// Slot for selecting an existing shape.
   TGeoShape *shape = fSelectedShape;
   new TGeoShapeDialog(fBSelShape, gClient->GetRoot(), 200,300);  
   fSelectedShape = (TGeoShape*)TGeoShapeDialog::GetSelected();
   if (fSelectedShape) fLSelShape->SetText(fSelectedShape->GetName());
   else fSelectedShape = shape;
}

//______________________________________________________________________________
void TGeoManagerEditor::DoSelectShape2()
{
// Slot for selecting a shape for making a volume.
   TGeoShape *shape = fSelectedShape2;
   new TGeoShapeDialog(fBSelShape2, gClient->GetRoot(), 200,300);  
   fSelectedShape2 = (TGeoShape*)TGeoShapeDialog::GetSelected();
   if (fSelectedShape2) fLSelShape2->SetText(fSelectedShape2->GetName());
   else fSelectedShape2 = shape;
}

//______________________________________________________________________________
void TGeoManagerEditor::DoSelectMaterial()
{
// Slot for selecting an existing material.
   TGeoMaterial *mat = fSelectedMaterial;
   new TGeoMaterialDialog(fBSelMaterial, gClient->GetRoot(), 200,300);  
   fSelectedMaterial = (TGeoMaterial*)TGeoMaterialDialog::GetSelected();
   if (fSelectedMaterial) fLSelMaterial->SetText(fSelectedMaterial->GetName());
   else fSelectedMaterial = mat;
}

//______________________________________________________________________________
void TGeoManagerEditor::DoSelectMaterial2()
{
// Slot for selecting an existing material and making a medium.
   TGeoMaterial *mat = fSelectedMaterial2;
   new TGeoMaterialDialog(fBSelMaterial2, gClient->GetRoot(), 200,300);  
   fSelectedMaterial2 = (TGeoMaterial*)TGeoMaterialDialog::GetSelected();
   if (fSelectedMaterial2) fLSelMaterial2->SetText(fSelectedMaterial2->GetName());
   else fSelectedMaterial2 = mat;
}

//______________________________________________________________________________
void TGeoManagerEditor::DoSelectMedium()
{
// Slot for selecting an existing medium.
   TGeoMedium *med = fSelectedMedium;
   new TGeoMediumDialog(fBSelMedium, gClient->GetRoot(), 200,300);  
   fSelectedMedium = (TGeoMedium*)TGeoMediumDialog::GetSelected();
   if (fSelectedMedium) fLSelMedium->SetText(fSelectedMedium->GetName());
   else fSelectedMedium = med;
}

//______________________________________________________________________________
void TGeoManagerEditor::DoSelectMedium2()
{
// Slot for selecting an existing medium for making a volume.
   TGeoMedium *med = fSelectedMedium2;
   new TGeoMediumDialog(fBSelMedium2, gClient->GetRoot(), 200,300);  
   fSelectedMedium2 = (TGeoMedium*)TGeoMediumDialog::GetSelected();
   if (fSelectedMedium2) fLSelMedium2->SetText(fSelectedMedium2->GetName());
   else fSelectedMedium2 = med;
}

//______________________________________________________________________________
void TGeoManagerEditor::DoSelectVolume()
{
// Slot for selecting an existing volume.
   TGeoVolume *vol = fSelectedVolume;
   new TGeoVolumeDialog(fBSelVolume, gClient->GetRoot(), 200,300);  
   fSelectedVolume = (TGeoVolume*)TGeoVolumeDialog::GetSelected();
   if (fSelectedVolume) fLSelVolume->SetText(fSelectedVolume->GetName());
   else fSelectedVolume = vol;
}

//______________________________________________________________________________
void TGeoManagerEditor::DoSelectTopVolume()
{
// Slot for seting top geometry volume.
   TGeoVolume *vol = fGeometry->GetTopVolume();
   new TGeoVolumeDialog(fBSelTop, gClient->GetRoot(), 200,300);  
   fSelectedVolume = (TGeoVolume*)TGeoVolumeDialog::GetSelected();
   if (fSelectedVolume) fLSelTop->SetText(fSelectedVolume->GetName());
   else fSelectedVolume = vol;
   if (fSelectedVolume && (fSelectedVolume != vol)) fGeometry->SetTopVolume(fSelectedVolume);
}

//______________________________________________________________________________
void TGeoManagerEditor::DoCloseGeometry()
{
// Slot for closing the geometry.
   if (!fGeometry->IsClosed()) fGeometry->CloseGeometry();
   fCloseGeometry->SetEnabled(kFALSE);
}
   
//______________________________________________________________________________
void TGeoManagerEditor::ShowSelectShape(Bool_t show)
{
// Show/hide interface for shape selection.
   TGCompositeFrame *cont = (TGCompositeFrame*)fCategories->GetItem("Shapes")->GetContainer();
   if (show) cont->ShowFrame(f2);
   else      cont->HideFrame(f2);
}
   
//______________________________________________________________________________
void TGeoManagerEditor::ShowSelectVolume(Bool_t show)
{
// Show/hide interface for volume selection.
   TGCompositeFrame *cont = (TGCompositeFrame*)fCategories->GetItem("General")->GetContainer();
   if (show) cont->ShowFrame(f7);
   else      cont->HideFrame(f7);
   cont = (TGCompositeFrame*)fCategories->GetItem("Volumes")->GetContainer();
   if (show) cont->ShowFrame(f3);
   else      cont->HideFrame(f3);
}
   
//______________________________________________________________________________
void TGeoManagerEditor::ShowSelectMaterial(Bool_t show)
{
// Show/hide interface for material selection.
   TGCompositeFrame *cont = (TGCompositeFrame*)fCategories->GetItem("Materials")->GetContainer();
   if (show) cont->ShowFrame(f4);
   else      cont->HideFrame(f4);
}
   
//______________________________________________________________________________
void TGeoManagerEditor::ShowSelectMedium(Bool_t show)
{
// Show/hide interface for medium selection.
   TGCompositeFrame *cont = (TGCompositeFrame*)fCategories->GetItem("Media")->GetContainer();
   if (show) cont->ShowFrame(f5);
   else      cont->HideFrame(f5);
}
   
//______________________________________________________________________________
void TGeoManagerEditor::ShowSelectMatrix(Bool_t show)
{
// Show/hide interface for matrix selection.
   TGCompositeFrame *cont = (TGCompositeFrame*)fCategories->GetItem("Matrices")->GetContainer();
   if (show) cont->ShowFrame(f6);
   else      cont->HideFrame(f6);
}
