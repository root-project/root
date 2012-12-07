{
#ifdef ClingWorkAroundMissingImplicitAuto
  TProfile2D*a,*b;
#endif
  a = new TProfile2D();
  b = new TProfile2D(*a);
  b->Print();
}
 
