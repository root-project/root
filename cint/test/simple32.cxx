void func(const char*,void*){
   fprintf(stderr,"const char*,void*\n");
};
void func(void*, const char*){
   fprintf(stderr,"void*,const char*\n");
}

void simple() {
   const char *ca;
	const char *cb;
	char *a;
	char *b;
	func(ca,b);
	func(a,cb);
}
