/* @(#)root/srputils:$Id$ */
/*
 * Create a private SRP passwd file.
 */

#include <sys/types.h>
#include <time.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <pwd.h>
#include <string.h>

#ifndef P_
#if __STDC__
#define P_(x) x
#else
#define P_(x) ()
#endif
#endif
#define STRFCPY(A,B) \
   (strncpy((A), (B), sizeof(A) - 1), (A)[sizeof(A) - 1] = '\0')
extern int obscure P_((const char *, const char *, const struct passwd *));
extern char *Basename P_((char *str));
extern struct passwd *get_my_pwent P_((void));
extern char *xstrdup P_((const char *str));

#include "pwauth.h"
#include "pwio.h"
#include "getdef.h"


/* EPS STUFF */
#include "t_pwd.h"
static int do_update_eps = 0;
struct t_pw eps_passwd;

/*
 * Global variables
 */

static char *name;              /* The name of user whose password is being changed */
static char *myname;            /* The current user's name */
static char *Prog;              /* Program name */

static char r_tconf[256];       /* -c config file */
static char r_passwd[256];      /* -p passwd file */

/*
 * External identifiers
 */

extern char *crypt_make_salt();
#if !defined(__GLIBC__)
extern char *l64a();
#endif

extern int optind;              /* Index into argv[] for current option */
extern char *optarg;            /* Pointer to current option value */


#define SROOTDCONF ".srootdpass.conf"
#define SROOTDPASS ".srootdpass"

/*
 * #defines for messages.  This facilities foreign language conversion
 * since all messages are defined right here.
 */
#define USAGE "usage: %s [ -c config ] [ -p passwd ] [ name ]\n"
#define NEWPASSMSG \
"Enter the new password (minimum of %d, maximum of %d characters)\n\
Please use a combination of upper and lower case letters and numbers.\n"
#define NEWPASS "New password:"
#define NEWPASS2 "Re-enter new password:"
#define NOMATCH "They don't match; try again.\n"

#define EPSFAIL "Unable to update EPS password.\n"
#define NOEPSCONF "Warning: configuration file missing; please run 'tconf'\n"

#define WHOAREYOU "%s: Cannot determine your user name.\n"
#define UNKUSER "%s: Unknown user %s\n"
#define UNCHANGED "The password for %s is unchanged.\n"

#define PASSWARN \
   "\nWarning: weak password (enter it again to use it anyway).\n"

/*
 * usage - print command usage and exit
 */

static void usage(int status)
{
   fprintf(stderr, USAGE, Prog);
   exit(status);
}


/*
 * new_password - validate old password and replace with new
 */

static int new_password(const struct passwd *pw)
{
   char clear[128];             /* Pointer to clear text */
   char *cp;                    /* Pointer to getpass() response */
   char orig[128];              /* Original password */
   char pass[128];              /* New password */
   int i;                       /* Counter for retries */
   int warned;

   /*
    * Get the new password.  The user is prompted for the new password
    * and has five tries to get it right.  The password will be tested
    * for strength, unless it is the root user.  This provides an escape
    * for initial login passwords.
    */

   warned = 0;
   for (i = getdef_num("PASS_CHANGE_TRIES", 5); i > 0; i--) {
      t_getpass(clear, 128, NEWPASS);
      cp = clear;
      if (!cp) {
         bzero(orig, sizeof orig);
         return -1;
      }
      if (warned && strcmp(pass, cp) != 0)
         warned = 0;
      STRFCPY(pass, cp);
      bzero(cp, strlen(cp));

      if (!warned && !obscure(orig, pass, pw)) {
         printf(PASSWARN);
         warned++;
         continue;
      }
      if (!(cp = getpass(NEWPASS2))) {
         bzero(orig, sizeof orig);
         return -1;
      }
      if (strcmp(cp, pass))
         fprintf(stderr, NOMATCH);
      else {
         bzero(cp, strlen(cp));
         break;
      }
   }
   bzero(orig, sizeof orig);

   if (i == 0) {
      bzero(pass, sizeof pass);
      return -1;
   }

   /*
    * Encrypt the password, then wipe the cleartext password.
    */

   /* EPS STUFF */
   {
      struct t_conf *tc;
      struct t_confent *tcent;

      if ((tc = t_openconfbyname(r_tconf)) == NULL ||
          (tcent = t_getconflast(tc)) == NULL) {
         fprintf(stderr, NOEPSCONF);
         do_update_eps = 0;
      } else {
         do_update_eps = 1;
         t_makepwent(&eps_passwd, name, pass, NULL, tcent);
      }

      if (tc)
         t_closeconf(tc);
   }

   bzero(pass, sizeof pass);

   return 0;
}


/*
 * rpasswd - change a user's password file information
 *
 * This command controls the password file and commands which are
 *    used to modify it.
 *
 * The valid options are
 *
 * -c config file name (default $HOME/.srootdpass.conf)
 * -p passwd file name (default $HOME/.srootdpass)
 *
 * Exit status:
 * 0 - success
 * 1 - permission denied
 * 2 - invalid combination of options
 * 3 - unexpected failure, password file unchanged
 * 5 - password file busy, try again later
 * 6 - invalid argument to option
 */

int main(int argc, char **argv)
{
   int flag;                    /* Current option to process                                                                                                                                                                                                                                    */
   const struct passwd *pw;     /* Password file entry for user      */

   /*
    * Get the program name.  The program name is used as a
    * prefix to most error messages.
    */

   Prog = Basename(argv[0]);

   sprintf(r_tconf, "%s/%s", getenv("HOME"), SROOTDCONF);
   sprintf(r_passwd, "%s/%s", getenv("HOME"), SROOTDPASS);

   /*
    * The remaining arguments will be processed one by one and
    * executed by this command.  The name is the last argument
    * if it does not begin with a "-", otherwise the name is
    * determined from the environment and must agree with the
    * real UID.  Also, the UID will be checked for any commands
    * which are restricted to root only.
    */

   while ((flag = getopt(argc, argv, "c:p:")) != EOF) {

      switch (flag) {

      case 'c':
         STRFCPY(r_tconf, optarg);
         printf("r_tconf: %s\n", r_tconf);
         break;
      case 'p':
         STRFCPY(r_passwd, optarg);
         printf("r_passwd: %s\n", r_passwd);
         break;
      default:
         usage(6);
      }
   }

   /*
    * Now I have to get the user name.  The name will be gotten
    * from the command line if possible.  Otherwise it is figured
    * out from the environment.
    */

   pw = get_my_pwent();
   if (!pw) {
      fprintf(stderr, WHOAREYOU, Prog);
      exit(1);
   }
   myname = xstrdup(pw->pw_name);
   if (optind < argc)
      name = argv[optind];
   else
      name = myname;

   /*
    * Let the user know whose password is being changed.
    */
   if (new_password(pw)) {
      fprintf(stderr, UNCHANGED, name);
      exit(1);
   }

/* EPS STUFF */

   if (do_update_eps) {
      FILE *passfp;

      /* try and see if the file is there, else create it */

      if ((passfp = fopen(r_passwd, "r+")) == NULL)
         creat(r_passwd, 0400);
      else
         fclose(passfp);

      if (t_changepw(r_passwd, &(eps_passwd.pebuf)) < 0)
         fprintf(stderr, EPSFAIL);

   } else
      fprintf(stderr, EPSFAIL);

   exit(0);
}
