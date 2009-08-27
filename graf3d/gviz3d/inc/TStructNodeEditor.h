// @(#)root/gviz3d:$Id$
// Author: Tomasz Sosnicki   18/09/09

/*************************************************************************
* Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#ifndef ROOT_TStructNodeEditor
#define ROOT_TStructNodeEditor

#include <TGedFrame.h>

class TList;
class TStructNode;
class TStructNodeProperty;
class TGNumberEntry;
class TGLabel;
class TGTextEntry;
class TGColorSelect;

class TStructNodeEditor : public TGedFrame {

protected:
   TList               *fColors;                // Pointer to list with class colors 
   TStructNode         *fNode;                  // Pointer to node which is edited
   TGNumberEntry       *fMaxObjectsNumberEntry; // Sets maximum number of nodes on scene
   TGNumberEntry       *fMaxLevelsNumberEntry;  // Sets maximum number of visible levels on scene
   TGLabel             *fTypeName;              // Label with name of type
   TGTextEntry         *fNameEntry;             // Text entry with name of property
   TGColorSelect       *fColorSelect;           // Control to selec a color
   TStructNodeProperty *fSelectedPropert;       // Pointer to property associated with node

   TStructNodeProperty* FindNodeProperty(TStructNode* node);
   TStructNodeProperty* GetDefaultProperty();

public:
   TStructNodeEditor(TList* colors, const TGWindow *p = 0, Int_t width = 140, Int_t height = 30,
      UInt_t options = kChildFrame, Pixel_t back = GetDefaultFrameBackground());
   ~TStructNodeEditor(); 

   void  ApplyButtonSlot();
   void  ColorSelectedSlot(Pixel_t color);
   void  DefaultButtonSlot();
   void  SetModel(TObject* obj);
   void  Update(Bool_t resetCamera);
   void  Update();

   ClassDef(TStructNodeEditor,1) // GUI fo editing TStructNode
};
#endif // ROOT_TStructNodeEditor

