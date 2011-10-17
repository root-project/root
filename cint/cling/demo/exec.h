int printf(const char*,...);
int exec(int argc, char* argv[]) {
   printf("argc=%d, arg0 is '%s'\n", argc, argv[0]);
   return 0;
}
