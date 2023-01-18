{
// Fill out the code of the actual test
printf("%s\n", "Hi mom!");
printf("%*s\n", 2, "Hi mom!");

const char *hdr = "header";
int num[3];
num[0] = 1;
num[1] = 3;
num[2] = -1;
const char *ext = ".cxx";
char sname[128];
int i=0;
while(-1!=num[i]) {
   snprintf(sname,128,"%s%d%s",hdr,num[i],ext);
   printf("%s\n",sname); //ci(cpp,sname,dfile,macro);
     ++i;
}

return 0;
}
