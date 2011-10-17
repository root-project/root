   printf("Yeah\n");
.L ./libReent.so
.L ./call_interp.h
   call_interp("printf(\"recurse!\\n\");");
