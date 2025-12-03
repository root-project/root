Double_t make_tuple_draw()
{
   std::cout.width(7);
   std::cout.precision(2);
   std::cout << evt.val1 << ", " << evt.val2 << ", " << evt.val3 << std::endl;
   int n = trk.N;
   std::cout << n << std::endl;
   for(int i=0; i<n; ++i) {
      float f = trk.arr[i];
      std::cout << i << " " << (int)f << std::endl;
   }
   return evt.val1;
}
