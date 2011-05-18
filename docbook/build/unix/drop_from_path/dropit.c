/* dropit.c, originated at Fermilab */

/* Note: this program was deliberately written in a K&R style. */

#include <stdio.h>
#include <string.h>
#ifndef __APPLE__
#include <malloc.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>

static const char *command = "<Unknown>";

static char *safe = 0;

struct Cell_s {
   struct Cell_s *next;
   char *value;
};

static struct Cell_s *pathlist = 0;

static struct Cell_s *
make_Cell(const char *str)
{
   struct Cell_s *retval = (struct Cell_s *)malloc(sizeof(struct Cell_s));
   if(retval) {
      if(str) {
         char *value = (char *)malloc(strlen(str) + 1);
         if(value) {
            (void) strcpy(value,str);
            retval->next = 0;
            retval->value = value;
         } else {
            free(retval);
            retval = 0;
         }
      }
   }
   return retval;
}

static void
destroy_Cell(
             struct Cell_s *cell)
{
   /* 
    ** This used to not be ifdefed, and sometimes free-ed
    ** things that were not malloc-ed. (like the "." string,
    ** and argv[n], etc.
    ** besides, we free a bunch of stuff and then exit, so
    ** why waste the cygles? -- mengel
    */
#ifdef slow_me_down_and_free_things_not_malloced
   if(cell) {
      if(cell->value) free(cell->value); 
      free(cell);
   }
#else
   if (cell) {}; // avoid unused variable message.
#endif
}

static void
add_at_front(
             struct Cell_s **list, struct Cell_s *cell)
{
   if(list && cell) {
      cell->next = *list;
      *list = cell;
   } else {
      printf("%s\n",safe);
      exit(1);
   }
}

static void
add_at_back(
            struct Cell_s **list, struct Cell_s *cell)
{
   if(list && cell) {
      struct Cell_s *previous, *current;
      for(previous = 0, current = *list;
          current;
          previous = current, current = current->next) {
      }
      if(previous) {
         previous->next = cell;
      } else {
         *list = cell;
      }
      cell->next = 0;
   } else {
      printf("%s\n",safe);
      exit(1);
   }
   
}

static int
contains(char *field,char *test)
{
   char *place;
   int len = strlen(test);
   
   if(len == 0) {
      return 1;
   }
   while( (place = strchr(field,test[0])) ) {
      if(strncmp(place,test,len) == 0) {
         return 1;
      }
      field = place + 1;
   }
   return 0;
}

static int
anchored(char *field,char *test)
{
   int len = strlen(test);
   if(len == 0) {
      return 1;
   }
   if (strncmp(field,test,len) == 0) {
      return 1;
   }
   return 0;
}

static int
exact(char *field,char *test)
{
   if(strcmp(field,test) == 0) {
      return 1;
   }
   return 0;
}

int main(int argc,char **argv)
{
   extern char *getenv();
   int (*compare)() = 0;
   void (*insert)() = 0;
   const char *path = "";
   const char *odel = "";
   const char *null = "";
   char idel = 0;
   char **cpp;
   // char *temp;
   extern int getopt();
   extern char *optarg;
   extern int optind; // , opterr;
   int any = 0;
   int opt;
   int error = 0;
   // int length;
   // int Lodel;
   // extern int errno;
   // extern char *sys_errlis[];
   int Anchored = 0;
   int Exact = 0;
   int first = 0;
   int cshmode = 0;
   int setup = 0;
   int Safe = 0;
   int duplicates = 1;
   int protected = 0;
   int existance = 0;
   int Ddefault = 1;
   int Edefault = 1;
   int Pdefault = 1;
   int Sdefault = 1;
   int adefault = 1;
   int cdefault = 1;
   int edefault = 1;
   int fdefault = 1;
   int idefault = 1;
   int ndefault = 1;
   int odefault = 1;
   int pdefault = 1;
   int sdefault = 1;
   struct Cell_s *previous, *current;
   struct stat sb;
   
   if(argv && *argv) {
      char *cp = strrchr(argv[0],'/');
      command = cp ? (cp+1) : *argv;
   }
   
   while((opt = getopt(argc,argv,"i:d:n:p:aefsciDEPS?")) != -1) {
      switch(opt) {
         case 'D':
            duplicates = 0;
            if(Ddefault) Ddefault = 0; else error = 1;
            break;
         case 'E':
            existance = 1;
            if(Edefault) Edefault = 0; else error = 1;
            break;
         case 'P':
            protected = 1;
            if(Pdefault) Pdefault = 0; else error = 1;
            break;
         case 'S':
            Safe = 1;
            if(Sdefault) Sdefault = 0; else error = 1;
            break;
         case 'a':
            Anchored = 1;
            if(adefault) adefault = 0; else error = 1;
            break;
         case 'e':
            Exact = 1;
            if(edefault) edefault = 0; else error = 1;
            break;
         case 'f':
            first = 1;
            if(fdefault) fdefault = 0; else error = 1;
            break;
         case 'p':
            path = optarg;
            if(pdefault) pdefault = 0; else error = 1;
            break;
         case 'i':
            idel = optarg[0];
            if(idefault) idefault = 0; else error = 1;
            break;
         case 'd':
            odel = optarg;
            if(odefault) odefault = 0; else error = 1;
            break;
         case 'n':
            null = optarg;
            if(ndefault) ndefault = 0; else error = 1;
            break;
         case 's':
            setup = 1;
            if(sdefault) sdefault = 0; else error = 1;
            break;
         case 'c':
            cshmode = 1;
            if(cdefault) cdefault = 0; else error = 1;
            break;
         default:
            error = 1;
      }
   }
   
   if(cshmode) {
      if(idefault) idel = pdefault ? ':' : ' ';
      if(odefault) odel = " ";
   } else {
      if(idefault) idel = ':';
      if(odefault) odel = ":";
   }
   
   /* Create a `safe' path, preferably their current PATH */
   
   {   char *syspath = getenv("PATH");
      if(!syspath) syspath = "/bin:/sbin:/usr/bin:/usr/bin";
      
      if(strlen(odel) == 1) {
         safe = (char *)malloc(strlen(syspath)+1);
         if(safe) {
            char *cp;
            (void) strcpy(safe,syspath);
            for(cp = safe; *cp; ++cp) if(*cp == ':') *cp = odel[0];
         }
      } else {
         char *cp;
         int del = 0;
         for(cp = syspath; *cp; ++cp) if(*cp == ':') ++del;
         safe = (char *)malloc(strlen(syspath)+del*(strlen(odel)-1)+1);
         if(safe) {
            char byte[2];
            byte[1] = '\0';
            safe[0] = '\0';
            for(cp = syspath; *cp; ++cp) {
               byte[0] = *cp;
               if(*cp == ':') {
                  strcat(safe,odel);
               } else {
                  strcat(safe,byte);
               }
            }
         }
      }
      if(!safe) {
         if(strlen(odel) == 1) {
            if(odel[0] == ' ') {
               safe = "/bin /sbin /usr/bin /usr/bin";
            } else if(odel[0] == ':') {
               safe = "/bin:/sbin:/usr/bin:/usr/bin";
            } else {
               safe = "/bin";
            }
         } else {
            safe = "/bin";
         }
      }
   }
   
   /* After this point, `safe' must not be changed! */
   
   if(error) {
      fprintf(stderr,
              "%s: usage: %s [-[aefscDEPS]] ... [-i x] [-d x] [-p path] [string] ...\n",
              command,command);
      printf("%s\n",safe);
      exit(1);
   }
   if(pdefault) ndefault = 1;
   if(ndefault) null = ".";
   
   if(pdefault) {
      path = getenv("PATH");
      if(!path) {
         idel = ':';
         path = "/bin:/sbin:/usr/bin:/usr/bin";
      }
   }
   
   if(path[0]) {
      char *lcp, *rcp;
      char *copy = (char *)malloc(strlen(path)+1);
      if(copy) {
         (void) strcpy(copy,path);
      } else {
         fprintf(stderr,"No memory to copy path\n");
         printf("%s\n",safe);
         exit(1);
      }
      for(lcp = copy, rcp = lcp; *rcp; ++rcp) {
         if(*rcp == idel) {
            *rcp = '\0';
            if(*lcp) {
               add_at_back(&pathlist,make_Cell(lcp));
            } else {
               add_at_back(&pathlist,make_Cell(null));
            }
            lcp = rcp + 1;
         }
      }
      if(*lcp) {
         add_at_back(&pathlist,make_Cell(lcp));
      } else {
         add_at_back(&pathlist,make_Cell(null));
      }
      free(copy);
   }
   
   compare = (Exact ? exact : (Anchored ? anchored : contains));
   insert = (first ? add_at_front : add_at_back);
   
   /* Do removals */
   for(cpp = argv+optind; *cpp; ++cpp) {
      struct Cell_s *next;
      if(Safe) {
         if(**cpp == '\0') continue;
         if(strcmp(*cpp,"/bin") == 0) continue;
      }
      for(previous = 0, current = pathlist; current; current = next) {
         next = current->next;
         if(compare(current->value,*cpp)) {
            if(previous) {
               previous->next = current->next;
            } else {
               pathlist = current->next;
            }
            destroy_Cell(current);
         } else {
            previous = current;
         }
      }
   }
   
   /* Do insertations, if requested */
   if(setup) {
      for(cpp = argv+optind; *cpp; ++cpp) {
         insert(&pathlist,
                make_Cell((**cpp == '\0') ? null : *cpp));
      }
   }
   
   /* Do insertations of system director(y)(ies), if requested */
   if(protected) {
      add_at_front(&pathlist,make_Cell("/usr/bin/X11"));
      add_at_front(&pathlist,make_Cell("/usr/etc"));
      add_at_front(&pathlist,make_Cell("/etc"));
      add_at_front(&pathlist,make_Cell("/usr/sbin"));
      add_at_front(&pathlist,make_Cell("/usr/bin"));
      add_at_front(&pathlist,make_Cell("/sbin"));
      add_at_front(&pathlist,make_Cell("/bin"));
      add_at_back(&pathlist,make_Cell("/usr/local/bin"));
   }
   
   /* Remove duplicates */
   if(Safe || protected || !duplicates) {
      for(current = pathlist; current; current = current -> next) {
         struct Cell_s *prev, *curr, *next;
         for(prev = current, curr = current->next; curr; curr = next) {
            next = curr->next;
            if(strcmp(current->value,curr->value) == 0) {
               prev->next = curr->next;
               destroy_Cell(curr);
            } else {
               prev = curr;
            }
         }
      }
   }
   
   /* Remove nonexistant file system objects or non-directories, */
   /* if requested */
   if(existance) {
      struct Cell_s *next;
      for(previous = 0, current = pathlist;
          current;
          current = current -> next) {
         next = current->next;
         sb.st_mode = -1;
         if(stat(current->value,&sb) ||
            (sb.st_mode & S_IFMT) != S_IFDIR) {
            if(previous) {
               previous->next = current->next;
            } else {
               pathlist = current->next;
            }
            destroy_Cell(current); 
         } else {
            previous = current;
         }
      }
   }
   
   /* Do insertations of minimal system director(y)(ies), */
   /* if requested and needed */
   if(Safe && !pathlist) {
      add_at_front(&pathlist,make_Cell("/usr/sbin"));
      add_at_front(&pathlist,make_Cell("/usr/bin"));
      add_at_front(&pathlist,make_Cell("/sbin"));
      add_at_front(&pathlist,make_Cell("/bin"));
      /* Remove nonexistant file system objects or non-directories */
      struct Cell_s *next;
      for(previous = 0, current = pathlist;
          current;
          current = current -> next) {
         next = current->next;
         sb.st_mode = -1;
         if(stat(current->value,&sb) ||
            (sb.st_mode & S_IFMT) != S_IFDIR) {
            if(previous) {
               previous->next = current->next;
            } else {
               pathlist = current->next;
            }
            destroy_Cell(current);
         } else {
            previous = current;
         }
      }
      /* Recheck */
      if(!pathlist) add_at_front(&pathlist,make_Cell("/bin"));
   }
   
   /* Generate the new PATH value */
   {
      struct Cell_s *next;
      for(current = pathlist; current; current = next) {
         next = current->next;
         if(any) printf("%s",odel); else any = 1;
         printf("%s",current->value);
         destroy_Cell(current);
      }
   }
   pathlist = 0;
   printf("\n");
   
   exit(0);
}

