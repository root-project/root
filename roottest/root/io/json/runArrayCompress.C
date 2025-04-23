#include "TBufferJSON.h"
#include "TArrayI.h"

bool testReading(TArrayI &arr, TString &json)
{
   TArrayI *arr2 = nullptr;

   TBufferJSON::FromJSON(arr2, json);
   if (!arr2) {
      cout << "Fail to read array from JSON" << endl;
      return false;
   }
   if (arr2->GetSize() != arr.GetSize()) {
      cout << "Array sizes mismatch " << arr.GetSize() << " != " << arr2->GetSize() << endl;
      delete arr2;
      return false;
   }

   for (int n=0;n<arr.GetSize();++n)
     if (arr.At(n) != arr2->At(n)) {
      cout << "Array content mismatch indx=" << n << "  " <<  arr.At(n) << " != " << arr2->At(n) << endl;
      delete arr2;
      return false;
   }

   delete arr2;
   return true;
}


void runArrayCompress()
{
   // Check creation of JSON for object, where TObject is not first parent
   // Also verify instrumented custom streamer of TPolyMarker3D

   TArrayI arr(100);
   for (Int_t n=0;n<arr.GetSize();++n) arr[n] = n;

   cout << "Plain and compressed array" << endl;
   TString json = TBufferJSON::ToJSON(&arr, 3);
   cout << json << endl;
   if (!testReading(arr, json)) return;

   json = TBufferJSON::ToJSON(&arr, 23);
   cout << json << endl;
   if (!testReading(arr, json)) return;

   arr.Reset(0);

   cout << "Empty array" << endl;
   json = TBufferJSON::ToJSON(&arr, 3);
   cout << json << endl;
   if (!testReading(arr, json)) return;
   json = TBufferJSON::ToJSON(&arr, 23);
   cout << json << endl;
   if (!testReading(arr, json)) return;

   for (Int_t n=0;n<10;++n) { arr[n+17] = 7; arr[n+56] = 11; }

   cout << "Array with many similar values" << endl;
   json = TBufferJSON::ToJSON(&arr, 3);
   cout << json << endl;
   if (!testReading(arr, json)) return;
   json = TBufferJSON::ToJSON(&arr, 23);
   cout << json << endl;
   if (!testReading(arr, json)) return;

   arr.Reset(0);
   for (Int_t n=0;n<10;++n) { arr[n+10] = n; arr[n+20] = 55; arr[n+30] = n; }

   cout << "Similar values inside" << endl;
   json = TBufferJSON::ToJSON(&arr, 3);
   cout << json << endl;
   if (!testReading(arr, json)) return;
   json = TBufferJSON::ToJSON(&arr, 23);
   cout << json << endl;
   if (!testReading(arr, json)) return;

   arr.Reset(0);
   for (Int_t n=0;n<10;++n) { arr[n+10] = 11; arr[n+20] = n+7; arr[n+30] = 22; }
   cout << "Similar values outside" << endl;
   json = TBufferJSON::ToJSON(&arr, 3);
   cout << json << endl;
   if (!testReading(arr, json)) return;
   json = TBufferJSON::ToJSON(&arr, 23);
   cout << json << endl;
   if (!testReading(arr, json)) return;

}
