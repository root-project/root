#include "TPolyMarker3D.h"
#include "TBufferJSON.h"
#include "TString.h"

void runPolyMarker()
{
   // Check creation of JSON for object, where TObject is not first parent
   // Also verify instrumented custom streamer of TPolyMarker3D

   TPolyMarker3D marker(10);
   for (Int_t n=0;n<10;++n)
      marker.SetPoint(n, n*2, n*3, n*4);

   TString json = TBufferJSON::ToJSON(&marker);

   std::cout << json << std::endl;
}
