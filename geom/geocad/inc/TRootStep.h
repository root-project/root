#ifndef ROOT_TRootStep_H
#define ROOT_TRootStep_H 1

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TGeoManager;
class OCCStep;

class  TRootStep: public TObject {

protected:

   TGeoManager *fGeometry; //ROOT geometry pointer
   OCCStep *fCreate; //OCC geometry build based on Root one

public:

   TRootStep(); 
   TRootStep(TGeoManager *geom);
   ~TRootStep();
   void *CreateGeometry();

   ClassDef(TRootStep,0); 
};
#endif
