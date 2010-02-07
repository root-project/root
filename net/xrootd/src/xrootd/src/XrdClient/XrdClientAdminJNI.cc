//         $Id$

//const char *XrdClientAdminJNICVSID = "$Id$";
#include "XrdClientAdminJNI.h"
#include "XrdClient/XrdClientAdmin.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdOuc/XrdOucString.hh"

#define DBGLVL 3
#define SETDEBUG EnvPutInt("DebugLevel", DBGLVL);

extern "C" JNIEXPORT jobject JNICALL Java_XrdClientAdminJNI_locate(JNIEnv *env,
                                                                jobject jobj,
                                                                jstring pathfile,
                                                                jstring hostname) {

   SETDEBUG

   // We want to get the value of firsturl
   jclass cls = env->GetObjectClass(jobj);
   jfieldID fid;
   jstring jstr;

   fid = env->GetFieldID(cls, "firsturl", "Ljava/lang/String;");
   if (fid == 0) {
      return (jobject )(new jboolean(false));
   }

   jstr = (jstring)env->GetObjectField(jobj, fid);
   const char *fu = env->GetStringUTFChars(jstr, 0);

   printf("firsturl: %s\n", fu);
   XrdClientAdmin *xrda = new XrdClientAdmin(fu);

   env->ReleaseStringUTFChars(jstr, fu);

   // Now we get the filename passed as parameter.
   kXR_char *filename = (kXR_char *)env->GetStringUTFChars(pathfile, 0);
   XrdClientUrlInfo finalloc;

   // Hence we locate that file
   // the answer is in finalloc
   bool r = false;
   if (xrda->Connect()) {
      r = xrda->Locate(filename, finalloc);
   }

   env->ReleaseStringUTFChars(pathfile, (const char *)filename);

   delete xrda;

   if (r)
      hostname = env->NewStringUTF(finalloc.Host.c_str());

   return (jobject)(new jboolean(r));
}

extern "C" JNIEXPORT jobject JNICALL Java_XrdClientAdminJNI_stat(JNIEnv *env,
                                                              jobject jobj,
                                                              jstring pathfile,
                                                              jint id,
                                                              jlong size,
                                                              jint flags,
                                                              jint modtime) {

   SETDEBUG

   // We want to get the value of firsturl
   jclass cls = env->GetObjectClass(jobj);
   jfieldID fid;
   jstring jstr;

   fid = env->GetFieldID(cls, "firsturl", "Ljava/lang/String;");
   if (fid == 0) {
      return (jobject)(new jboolean(false));
   }

   jstr = (jstring)env->GetObjectField(jobj, fid);
   const char *fu = env->GetStringUTFChars(jstr, 0);

   printf("firsturl: %s\n", fu);
   XrdClientAdmin *xrda = new XrdClientAdmin(fu);

   env->ReleaseStringUTFChars(jstr, fu);

   // Now we get the filename passed as parameter.
   kXR_char *filename = (kXR_char *)env->GetStringUTFChars(pathfile, 0);

   // Hence we stat that file
   bool r = false;
   long myid, myflags, mymodtime;
   long long mysize;
   if (xrda->Connect()) {
      r = xrda->Stat((const char*)filename, myid, mysize, myflags, mymodtime);
   }

   env->ReleaseStringUTFChars(pathfile, (const char *)filename);

   delete xrda;
   

   if (r) {
      // Here we copy the results into the java parameters
      id = myid;
      size = mysize;
      flags = myflags;
      modtime = mymodtime;
   }

   return (jobject)(new jboolean(r));


}

JNIEXPORT jobject JNICALL Java_XrdClientAdminJNI_chmod(JNIEnv *env,
                                                       jobject jobj,
                                                       jstring pathfile, 
                                                       jint user,
                                                       jint group,
                                                       jint other) {


   SETDEBUG

   // We want to get the value of firsturl
   jclass cls = env->GetObjectClass(jobj);
   jfieldID fid;
   jstring jstr;

   fid = env->GetFieldID(cls, "firsturl", "Ljava/lang/String;");
   if (fid == 0) {
      return (jobject)(new jboolean(false));
   }

   jstr = (jstring)env->GetObjectField(jobj, fid);
   const char *fu = env->GetStringUTFChars(jstr, 0);

   printf("firsturl: %s\n", fu);
   XrdClientAdmin *xrda = new XrdClientAdmin(fu);

   env->ReleaseStringUTFChars(jstr, fu);

   // Now we get the filename passed as parameter.
   kXR_char *filename = (kXR_char *)env->GetStringUTFChars(pathfile, 0);

   // Hence we chmod that file
   bool r = false;
   if (xrda->Connect()) {
      r = xrda->Chmod((const char*)filename, user, group, other);
   }

   env->ReleaseStringUTFChars(pathfile, (const char *)filename);

   delete xrda;


   return (jobject)(new jboolean(r));

}


JNIEXPORT jobject JNICALL Java_XrdClientAdminJNI_dirlist(JNIEnv *env,
                                                         jobject jobj,
                                                         jstring path,
                                                         jobjectArray result) {




   SETDEBUG

   // We want to get the value of firsturl
   jclass cls = env->GetObjectClass(jobj);
   jfieldID fid;
   jstring jstr;

   fid = env->GetFieldID(cls, "firsturl", "Ljava/lang/String;");
   if (fid == 0) {
      return (jobject)(new jboolean(false));
   }

   jstr = (jstring)env->GetObjectField(jobj, fid);
   const char *fu = env->GetStringUTFChars(jstr, 0);

   printf("firsturl: %s\n", fu);
   XrdClientAdmin *xrda = new XrdClientAdmin(fu);

   env->ReleaseStringUTFChars(jstr, fu);

   // Now we get the filename passed as parameter.
   kXR_char *filename = (kXR_char *)env->GetStringUTFChars(path, 0);

   // Hence we dirlist that path
   bool r = false;
   vecString vs;
   if (xrda->Connect()) {
      r = xrda->DirList((const char*)filename, vs);
   }

   env->ReleaseStringUTFChars(path, (const char *)filename);

   delete xrda;

   if (r && vs.GetSize()) {
      // If the request went good, we copy the results to the java env
      result = (jobjectArray)env->NewObjectArray( vs.GetSize(), env->FindClass("java/lang/String"),
         env->NewStringUTF("") );

      for(int i = 0; i < vs.GetSize(); i++)
         env->SetObjectArrayElement(
            result, i, env->NewStringUTF(vs[i].c_str()) );
   }

   return (jobject)(new jboolean(r));

}

JNIEXPORT jobject JNICALL Java_XrdClientAdminJNI_existfiles(JNIEnv *env,
                                                            jobject jobj,
                                                            jobjectArray filez,
                                                            jobjectArray xstfilez) {


   SETDEBUG

   // We want to get the value of firsturl
   jclass cls = env->GetObjectClass(jobj);
   jfieldID fid;
   jstring jstr;

   fid = env->GetFieldID(cls, "firsturl", "Ljava/lang/String;");
   if (fid == 0) {
      return (jobject)(new jboolean(false));
   }

   jstr = (jstring)env->GetObjectField(jobj, fid);
   const char *fu = env->GetStringUTFChars(jstr, 0);

   printf("firsturl: %s\n", fu);
   XrdClientAdmin *xrda = new XrdClientAdmin(fu);

   env->ReleaseStringUTFChars(jstr, fu);

   // Now we build the request list from the one coming
   vecString vs;
   for (int i = 0; i < env->GetArrayLength(filez); i++) {
      jstring jstr;
      const char *str;
      XrdOucString s;

      jstr = (jstring)env->GetObjectArrayElement(filez, i);
      str = env->GetStringUTFChars(jstr, 0);
      s = str;
      env->ReleaseStringUTFChars(jstr, str);

      vs.Push_back(s);

   }


   // Hence we existfiles that list
   bool r = false;
   vecBool vb;
   if (xrda->Connect()) {
      r = xrda->ExistFiles(vs, vb);
   }

   delete xrda;

   if (r && vb.GetSize()) {
      // If the request went good, we copy the results to the java env
      xstfilez = (jobjectArray)env->NewBooleanArray( vb.GetSize() );

      jboolean jb;
      for(int i = 0; i < vb.GetSize(); i++) {
         jb = vb[i];
         env->SetObjectArrayElement(
            xstfilez, i, (jobject)(new jboolean(jb)) );
      }


   }

   return (jobject)(new jboolean(r));

}


JNIEXPORT jobject JNICALL Java_XrdClientAdminJNI_existdirs(JNIEnv *env,
                                                           jobject jobj,
                                                           jobjectArray dirz,
                                                           jobjectArray xstdirz) {


   SETDEBUG

   // We want to get the value of firsturl
   jclass cls = env->GetObjectClass(jobj);
   jfieldID fid;
   jstring jstr;

   fid = env->GetFieldID(cls, "firsturl", "Ljava/lang/String;");
   if (fid == 0) {
      return (jobject)(new jboolean(false));
   }

   jstr = (jstring)env->GetObjectField(jobj, fid);
   const char *fu = env->GetStringUTFChars(jstr, 0);

   printf("firsturl: %s\n", fu);
   XrdClientAdmin *xrda = new XrdClientAdmin(fu);

   env->ReleaseStringUTFChars(jstr, fu);

   // Now we build the request list from the one coming
   vecString vs;
   for (int i = 0; i < env->GetArrayLength(dirz); i++) {
      jstring jstr;
      const char *str;
      XrdOucString s;

      jstr = (jstring)env->GetObjectArrayElement(dirz, i);
      str = env->GetStringUTFChars(jstr, 0);
      s = str;
      env->ReleaseStringUTFChars(jstr, str);

      vs.Push_back(s);

   }


   // Hence we existdirs that list
   bool r = false;
   vecBool vb;
   if (xrda->Connect()) {
      r = xrda->ExistDirs(vs, vb);
   }

   delete xrda;

   if (r && vb.GetSize()) {
      // If the request went good, we copy the results to the java env
      xstdirz = (jobjectArray)env->NewBooleanArray( vb.GetSize() );

      jboolean jb;
      for(int i = 0; i < vb.GetSize(); i++) {
         jb = vb[i];
         env->SetObjectArrayElement(
            xstdirz, i, (jobject )(new jboolean(jb)) );
      }


   }

   return (jobject)(new jboolean(r));

}

JNIEXPORT jobject JNICALL Java_XrdClientAdminJNI_getchecksum(JNIEnv *env,
                                                             jobject jobj,
                                                             jstring pathfile,
                                                             jstring chksum) {

   SETDEBUG

   // We want to get the value of firsturl
   jclass cls = env->GetObjectClass(jobj);
   jfieldID fid;
   jstring jstr;

   fid = env->GetFieldID(cls, "firsturl", "Ljava/lang/String;");
   if (fid == 0) {
      return (jobject)(new jboolean(false));
   }

   jstr = (jstring)env->GetObjectField(jobj, fid);
   const char *fu = env->GetStringUTFChars(jstr, 0);

   printf("firsturl: %s\n", fu);
   XrdClientAdmin *xrda = new XrdClientAdmin(fu);

   env->ReleaseStringUTFChars(jstr, fu);

   // Now we get the filename passed as parameter.
   kXR_char *filename = (kXR_char *)env->GetStringUTFChars(pathfile, 0);

   // Hence we chksum that file
   bool r = false;
   kXR_char *chksumbuf = 0;

   if (xrda->Connect()) {
      r = xrda->GetChecksum((kXR_char *)filename, &chksumbuf);
   }

   env->ReleaseStringUTFChars(pathfile, (const char *)filename);

   delete xrda;


   if (r) {
      chksum = env->NewStringUTF( (const char *)chksumbuf );
      delete chksumbuf;
   }

   return (jobject)(new jboolean(r));

}

JNIEXPORT jobject JNICALL Java_XrdClientAdminJNI_isfileonline(JNIEnv *env,
                                                              jobject jobj,
                                                              jobjectArray filez,
                                                              jobjectArray flzonline) {



   SETDEBUG

   // We want to get the value of firsturl
   jclass cls = env->GetObjectClass(jobj);
   jfieldID fid;
   jstring jstr;

   fid = env->GetFieldID(cls, "firsturl", "Ljava/lang/String;");
   if (fid == 0) {
      return (jobject)(new jboolean(false));
   }

   jstr = (jstring)env->GetObjectField(jobj, fid);
   const char *fu = env->GetStringUTFChars(jstr, 0);

   printf("firsturl: %s\n", fu);
   XrdClientAdmin *xrda = new XrdClientAdmin(fu);

   env->ReleaseStringUTFChars(jstr, fu);

   // Now we build the request list from the one coming
   vecString vs;
   for (int i = 0; i < env->GetArrayLength(filez); i++) {
      jstring jstr;
      const char *str;
      XrdOucString s;

      jstr = (jstring)env->GetObjectArrayElement(filez, i);
      str = env->GetStringUTFChars(jstr, 0);
      s = str;
      env->ReleaseStringUTFChars(jstr, str);

      vs.Push_back(s);

   }

   // Hence we isfileonline that list
   bool r = false;
   vecBool vb;
   if (xrda->Connect()) {
      r = xrda->IsFileOnline(vs, vb);
   }

   delete xrda;

   if (r && vb.GetSize()) {
      // If the request went good, we copy the results to the java env
      flzonline = (jobjectArray)env->NewBooleanArray( vb.GetSize() );

      jboolean jb;
      for(int i = 0; i < vb.GetSize(); i++) {
         jb = vb[i];
         env->SetObjectArrayElement(
            flzonline, i, (jobject )(new jboolean(jb)) );
      }

   }

   return (jobject)(new jboolean(r));

}



JNIEXPORT jobject JNICALL Java_XrdClientAdminJNI_mv(JNIEnv *env,
                                                    jobject jobj,
                                                    jstring pathname1,
                                                    jstring pathname2) {


   SETDEBUG

   // We want to get the value of firsturl
   jclass cls = env->GetObjectClass(jobj);
   jfieldID fid;
   jstring jstr;

   fid = env->GetFieldID(cls, "firsturl", "Ljava/lang/String;");
   if (fid == 0) {
      return (jobject)(new jboolean(false));
   }

   jstr = (jstring)env->GetObjectField(jobj, fid);
   const char *fu = env->GetStringUTFChars(jstr, 0);

   printf("firsturl: %s\n", fu);
   XrdClientAdmin *xrda = new XrdClientAdmin(fu);

   env->ReleaseStringUTFChars(jstr, fu);

   // Now we get the filenames passed as parameters.
   kXR_char *filename1 = (kXR_char *)env->GetStringUTFChars(pathname1, 0);
   kXR_char *filename2 = (kXR_char *)env->GetStringUTFChars(pathname2, 0);

   // Hence we chmod that file
   bool r = false;
   if (xrda->Connect()) {
      r = xrda->Mv((const char*)filename1, (const char*)filename2);
   }

   env->ReleaseStringUTFChars(pathname1, (const char *)filename1);
   env->ReleaseStringUTFChars(pathname2, (const char *)filename2);

   delete xrda;

   return (jobject)(new jboolean(r));



};



JNIEXPORT jobject JNICALL Java_XrdClientAdminJNI_mkdir(JNIEnv *env,
                                                       jobject jobj,
                                                       jstring pathname,
                                                       jint user, jint group, jint other) {


   SETDEBUG

   // We want to get the value of firsturl
   jclass cls = env->GetObjectClass(jobj);
   jfieldID fid;
   jstring jstr;

   fid = env->GetFieldID(cls, "firsturl", "Ljava/lang/String;");
   if (fid == 0) {
      return (jobject)(new jboolean(false));
   }

   jstr = (jstring)env->GetObjectField(jobj, fid);
   const char *fu = env->GetStringUTFChars(jstr, 0);

   printf("firsturl: %s\n", fu);
   XrdClientAdmin *xrda = new XrdClientAdmin(fu);

   env->ReleaseStringUTFChars(jstr, fu);

   // Now we get the filename passed as parameter.
   kXR_char *filename = (kXR_char *)env->GetStringUTFChars(pathname, 0);

   // Hence we mkdir that path
   bool r = false;
   if (xrda->Connect()) {
      r = xrda->Mkdir((const char*)filename, user, group, other);
   }

   env->ReleaseStringUTFChars(pathname, (const char *)filename);

   delete xrda;


   return (jobject)(new jboolean(r));




}




JNIEXPORT jobject JNICALL Java_XrdClientAdminJNI_rm(JNIEnv *env,
						    jobject jobj,
						    jstring fname) {




   SETDEBUG

   // We want to get the value of firsturl
   jclass cls = env->GetObjectClass(jobj);
   jfieldID fid;
   jstring jstr;

   fid = env->GetFieldID(cls, "firsturl", "Ljava/lang/String;");
   if (fid == 0) {
      return (jobject)(new jboolean(false));
   }

   jstr = (jstring)env->GetObjectField(jobj, fid);
   const char *fu = env->GetStringUTFChars(jstr, 0);

   printf("firsturl: %s\n", fu);
   XrdClientAdmin *xrda = new XrdClientAdmin(fu);

   env->ReleaseStringUTFChars(jstr, fu);

   // Now we get the filename passed as parameter.
   kXR_char *filename = (kXR_char *)env->GetStringUTFChars(fname, 0);

   // Hence we chmod that file
   bool r = false;
   if (xrda->Connect()) {
      r = xrda->Rm((const char*)filename);
   }

   env->ReleaseStringUTFChars(fname, (const char *)filename);

   delete xrda;


   return (jobject)(new jboolean(r));



}




JNIEXPORT jobject JNICALL Java_XrdClientAdminJNI_rmdir(JNIEnv *env,
                                                       jobject jobj,
                                                       jstring pathname) {

   SETDEBUG

   // We want to get the value of firsturl
   jclass cls = env->GetObjectClass(jobj);
   jfieldID fid;
   jstring jstr;

   fid = env->GetFieldID(cls, "firsturl", "Ljava/lang/String;");
   if (fid == 0) {
      return (jobject)(new jboolean(false));
   }

   jstr = (jstring)env->GetObjectField(jobj, fid);
   const char *fu = env->GetStringUTFChars(jstr, 0);

   printf("firsturl: %s\n", fu);
   XrdClientAdmin *xrda = new XrdClientAdmin(fu);

   env->ReleaseStringUTFChars(jstr, fu);

   // Now we get the pathname passed as parameter.
   kXR_char *filename = (kXR_char *)env->GetStringUTFChars(pathname, 0);

   // Hence we rmdir that path
   bool r = false;
   if (xrda->Connect()) {
      r = xrda->Rmdir((const char*)filename);
   }

   env->ReleaseStringUTFChars(pathname, (const char *)filename);

   delete xrda;


   return (jobject)(new jboolean(r));


}

JNIEXPORT jobject JNICALL Java_XrdClientAdminJNI_prepare(JNIEnv *env,
                                                         jobject jobj,
                                                         jobjectArray filez,
                                                         jchar opts, jchar prio) {



   SETDEBUG

   // We want to get the value of firsturl
   jclass cls = env->GetObjectClass(jobj);
   jfieldID fid;
   jstring jstr;

   fid = env->GetFieldID(cls, "firsturl", "Ljava/lang/String;");
   if (fid == 0) {
      return (jobject)(new jboolean(false));
   }

   jstr = (jstring)env->GetObjectField(jobj, fid);
   const char *fu = env->GetStringUTFChars(jstr, 0);

   printf("firsturl: %s\n", fu);
   XrdClientAdmin *xrda = new XrdClientAdmin(fu);

   env->ReleaseStringUTFChars(jstr, fu);

   // Now we build the request list from the one coming
   vecString vs;
   for (int i = 0; i < env->GetArrayLength(filez); i++) {
      jstring jstr;
      const char *str;
      XrdOucString s;

      jstr = (jstring)env->GetObjectArrayElement(filez, i);
      str = env->GetStringUTFChars(jstr, 0);
      s = str;
      env->ReleaseStringUTFChars(jstr, str);

      vs.Push_back(s);

   }


   // Hence we prepare that list
   bool r = false;
   if (xrda->Connect()) {
      r = xrda->Prepare(vs, opts, prio);
   }

   delete xrda;

   return (jobject)(new jboolean(r));

}


