#if defined(TARGET)
const char *debug = TARGET;
#endif
int main() 
{
   return !printf("target is %s\n",debug);
}
