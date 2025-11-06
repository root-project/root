#ifndef GUARD__TRACK__
#define GUARD__TRACK__
class Track : public TObject {
public:
   int a;
   float bb[2];
   double c[2][3];
   void Set(int i= 3) {
      a = i;
      for(int j=0;j<2;++j) {
         bb[j] = i*100+j*10;
         for(int k=0;k<3;++k) {
            c[j][k] = i*100+j*10+k;
         }
      }
   }
   ClassDefOverride(Track,1);
};
#endif

