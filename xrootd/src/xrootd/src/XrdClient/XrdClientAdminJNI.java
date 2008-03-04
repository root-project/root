package xrootdadmin;

public class XrdClientAdminJNI {

    // This usually is an xrootd redirector
    String firsturl;


    public native boolean chmod(String locpathfile1, int user, int group, int other);

    public native boolean dirlist(String path, String[] lst);

    public native boolean existfiles(String[] pathslist, boolean[] res);

    public native boolean existdirs(String[] pathslist, boolean[] res);

    public native boolean getchecksum(String pathname, String chksum);

    public native boolean isfileonline(String[] pathslist, boolean[] res);

    // Finds one of the final locations for a given file
    // Returns false if errors occurred
    public native boolean locate(String pathfile, String hostname);

    public native boolean mv(String locpathfile1, String locpathfile2);

    public native boolean mkdir(String locpathfile1, int user, int group, int other);

    public native boolean rm(String locpathfile);

    public native boolean rmdir(String locpath);

    public native boolean prepare(String[] pathnamelist, char opts, char priority);

    // Gives info for a given file
    public native boolean stat(String pathfile, int id, long size, int flags, int modtime);



    public XrdClientAdminJNI(String hostname) { firsturl = "xroot://"+hostname+"//dummy"; };

    public static void main(String args[]) {
	XrdClientAdminJNI a = new XrdClientAdminJNI("kanolb-a.slac.stanford.edu");
	String newhost = "";
	boolean r = a.locate("pippo.root", newhost);
	System.out.println("Locate Result: " + r + " host: '" + newhost + "'");
    }

    static {
	System.loadLibrary("XrdClientAdminJNI");
    }
}
