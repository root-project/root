
Double_t val3dim()
{
   int ntrack = trs.GetEntries();
   fprintf(stdout,"ntracks = %d\n",ntrack);
   for(int i=0;i<2;++i) {
      fprintf(stdout,"trs[%d].a = %d\n",i,(int)trs.a[i]);
      for(int j=0;j<2;++j) {
         fprintf(stdout,"trs[%d].bb[%d] = %f\n",i,j,trs.bb[i][j]);
         for(int k=0;k<3;++k) {
            fprintf(stdout,"trs[%d].c[%d][%d] = %g\n",i,j,k,trs.c[i][j][k]);
         }
      }
   }

   fprintf(stdout,"tr.a = %d\n",(int)tr.a);
   for(int i=0;i<2;++i) {
      fprintf(stdout,"tr.bb[%d] = %f\n",i,tr.bb[i]);
      for(int j=0;j<3;++j) {
         fprintf(stdout,"tr.c[%d][%d] = %g\n",i,j,tr.c[i][j]);
      }
   }
   for(int i=0;i<2;++i) {
      fprintf(stdout,"c[%d] = %d\n",i,c[i]);
      for(int j=0;j<3;++j) {
         fprintf(stdout,"bb[%d][%d] = %g\n",i,j,bb[i][j]);
         for(int k=0;k<4;++k) {
            fprintf(stdout,"a[%d][%d][%d] = %g\n",i,j,k,a[i][j][k]);
         }
      }

   }
   double d1 = a[0][0][0];
   double d2 = bb[0][0];
   int c1 = c[0];

   return d1+d2+c1;
}

/*

   TMultiArrayProxy<TMultiArrayType<TArrayType<double,4>,3> >  a;
   TMultiArrayProxy<TArrayType<double,3> >  b;
   TMultiArrayProxy<TArrayType<int> >  c;

   */

