double pns() 
{
   printf("trajPos.fP.fX[1]=%3.2f\n",trajPos.fP.fX[1]); 
   printf("trajPos.fP[1].fX=%3.2f\n",trajPos.fP[1].X()); 
   printf("trajPos[1].fP.fX=%3.2f\n",trajPos[1].Vect().X()); 
   return 1; 
}
