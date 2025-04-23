struct RAII {
   RAII() { printf("RAII()\n"); }
   ~RAII() { printf("~RAII()\n"); }
} raiiObj;
void assertUnloadHelper() {
   printf("Called the helper!\n");
}
