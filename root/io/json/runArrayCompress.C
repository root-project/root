#include "TBufferJSON.h"
#include "TArrayI.h"

void runArrayCompress()
{
   // Check creation of JSON for object, where TObject is not first parent
   // Also verify instrumented custom streamer of TPolyMarker3D

   TArrayI arr(100);
   for (Int_t n=0;n<arr.GetSize();++n) arr[n] = n;

   cout << "Plain and compressed array" << endl;
   TString json = TBufferJSON::ToJSON(&arr, 3);
   cout << json << endl;
   json = TBufferJSON::ToJSON(&arr, 23);
   cout << json << endl;

   arr.Reset(0);

   cout << "Empty array" << endl;
   json = TBufferJSON::ToJSON(&arr, 3);
   cout << json << endl;
   json = TBufferJSON::ToJSON(&arr, 23);
   cout << json << endl;

   for (Int_t n=0;n<10;++n) { arr[n+17] = 7; arr[n+56] = 11; }

   cout << "Array with many similar values" << endl;
   json = TBufferJSON::ToJSON(&arr, 3);
   cout << json << endl;
   json = TBufferJSON::ToJSON(&arr, 23);
   cout << json << endl;

   arr.Reset(0);
   for (Int_t n=0;n<10;++n) { arr[n+10] = n; arr[n+20] = 55; arr[n+30] = n; }

   cout << "Similar values inside" << endl;
   json = TBufferJSON::ToJSON(&arr, 3);
   cout << json << endl;
   json = TBufferJSON::ToJSON(&arr, 23);
   cout << json << endl;

   arr.Reset(0);
   for (Int_t n=0;n<10;++n) { arr[n+10] = 11; arr[n+20] = n+7; arr[n+30] = 22; }
   cout << "Similar values outside" << endl;
   json = TBufferJSON::ToJSON(&arr, 3);
   cout << json << endl;
   json = TBufferJSON::ToJSON(&arr, 23);
   cout << json << endl;

}
