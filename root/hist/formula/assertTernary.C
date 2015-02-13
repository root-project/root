int assertTernary() {
   TF1 f6( "f6", "(x*[0])*((x<[1])?([2]*x):([3]+x))", 1, 10 );
   f6.SetParameters( 1, 4, 2, 4 );
   long eval = (long) f6.Eval(2);
   double x=2; 
   long calc = x*1*((x<4)?(2*x):(4+x));
   
   int result = 0;
   if (calc != eval) {
      fprintf(stderr,"TFormula does not properly evaluate ternary operator got %ld rather than %ld\n",eval,calc);
      ++result;
   }
   
   TF1 f1( "f1", "[0]*x", 1, 100 );
   TF1 f2( "f2", "[0]+x" ,1, 100 );
   TF1 f3( "f3", "(x<[0])?(f1):(f2)", 1, 100 );
   double par_a = 1; TF1 f4a( "f4a", "[0]", 1, 100);
   double par_b = 0; TF1 f4b( "f4b", "x**[0]", 1, 100 );
   TF1 f5a( "f5a", "(f3)*(f4a)", 1, 100 );
   TF1 f5b( "f5b", "(f3)*(f4b)", 1, 100 );
   
   f1.SetParameter( 0, 2 ); 
   f2.SetParameter( 0, 40 );
   f3.SetParameters( 40, 2, 40 );
   f4a.SetParameter( 0, par_a );
   f5a.SetParameters( 40, 2, 40, par_a );
   f4b.SetParameter( 0, par_b );
   f5b.SetParameters( 40, 2, 40, par_b );
   
   const int Nx = 2;
   double xs[ Nx ] = { 39, 41 };
   
   for( int ix=0; ix<Nx; ++ix ) {
      double x = xs[ ix ];
      cout<< x << "\ta: " << f1.Eval(x) <<", "<<f2.Eval(x)<<", "<<f3.Eval(x)<<",\t"<<f4a.Eval(x)<<" -> "<<f3.Eval(x)*f4a.Eval(x)<<" =? "<<f5a.Eval(x)<<endl;
      long eval1 = f3.Eval(x)*f4a.Eval(x);
      long eval2 = f5a.Eval(x);
      if (eval1 != eval2) {
         fprintf(stderr,"TFormula does not properly evaluate ternary operator for f3.Eval(x)*f4a.Eval(x) got %ld and for f5a.Eval(x) got %ld\n",eval1,eval2);
         ++result;
      }         
      cout<<"\tb:\t\t\t"<<f4b.Eval(x)<<" -> "<<f3.Eval(x)*f4b.Eval(x)<<" =? "<<f5b.Eval(x)<<endl;
      eval1 = f3.Eval(x)*f4b.Eval(x);
      eval2 = f5b.Eval(x);
      if (eval1 != eval2) {
         fprintf(stderr,"TFormula does not properly evaluate ternary operator for f3.Eval(x)*f4b.Eval(x) got %ld and for f5b.Eval(x) got %ld\n",eval1,eval2);
         ++result;
      }         
   }
   
   return result;
}
