{
#ifdef ClingWorkAroundMissingImplicitAuto
  TProfile2D*a,*b;
#endif
  a = new TProfile2D();

#ifdef ClingWorkAroundExtraParensWithImplicitAuto
  b = new TProfile2D(*a);
#endif
  b->Print();
}
 
