#include "TPolyMarker3D.h"
#include "TBufferXML.h"
#include "TString.h"

void runPolyMarker()
{
   // Check creation of XML for object, where TObject is not first parent
   // Also verify instrumented custom streamer of TPolyMarker3D

   TPolyMarker3D* marker = new TPolyMarker3D(10);
   for (Int_t n=0;n<10;++n)
      marker->SetPoint(n, n*2, n*3, n*4);

   marker->SetName("TestMarker");

   TString xml = TBufferXML::ConvertToXML(marker);

   cout << xml << endl;

   TPolyMarker3D* marker2 = dynamic_cast<TPolyMarker3D*> (TBufferXML::ConvertFromXML(xml));

   cout << "Read " << marker2->GetName() << " GetN = " << marker2->GetN() << endl;
}
