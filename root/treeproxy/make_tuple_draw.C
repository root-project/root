Double_t make_tuple_draw() {
   cout.width(7);
   cout << std::setprecision(2)  << evt.val1 << ", " << evt.val2 << ", " << evt.val3 << endl;
   int n = trk.N;
   cout << n << endl;
   for(int i=0; i<n; ++i) {
      float f = trk.arr[i];
      cout << i << " " << (int)f << endl;
   }
   return evt.val1;
}
