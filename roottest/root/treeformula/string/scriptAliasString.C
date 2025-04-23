

double scriptAliasString() {
   const char *str_name = name.c_str();
   printf("the string is %s\n",str_name);
   str_name = name;
   printf("the string with operator use is %s\n",str_name);
   printf("the string direct is %s\n",(const char*)name);
   return 0;
}

