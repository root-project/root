#include "TPolyMarker3D.h"
#include "TBufferXML.h"
#include "TString.h"

void runPolyMarker()
{
   // Check creation of XML for object, where TObject is not first parent
   // Also verify instrumented custom streamer of TPolyMarker3D

   TPolyMarker3D marker(10);
   for (Int_t n=0;n<10;++n)
      marker.SetPoint(n, n*2, n*3, n*4);

   marker.SetName("TestMarker");

   TString xml = TBufferXML::ConvertToXML(&marker);

   std::cout << xml << std::endl;

   auto marker2 = dynamic_cast<TPolyMarker3D*> (TBufferXML::ConvertFromXML(xml));

   if (marker2)
      std::cout << "Read " << marker2->GetName() << " GetN = " << marker2->GetN() << std::endl;
   else
      std::cerr << "Fail to read TPolyMarker3D from XML" << std::endl;
   delete marker2;
}
