ROOTQL
======

This is a Quick Look plugin that allows quick inspection of the content
of a ROOT file. 

Quick Look is available on MacOS X since version 10.5 (Leopard). To use QL
select a file icon in the Finder and hit the space bar. For all file types
supported by QL you will get a window showing the file content, for file types
not supported you will get a generic window showing some basic file info.

The idea of QL is that file content can be shown without the heavy application
startup process. Generating a QL view of a ROOT file depends on the size of the
file, but generally it is a quick operation.

Get the binary for the ROOTQL plugin from:

   ftp://root.cern.ch/root/ROOTQL.tgz

To install the plugin, after untarring the above file, just drag the bundle
ROOTQL.qlgenerator to /Library/QuickLook (global, i.e. for all users on a
system) or to ~/Library/QuickLook (local, this user only) directory.
You may need to create that folder if it doesn't already exist.

To build from source, get it from svn using:

   svn co http://root.cern.ch/svn/root/trunk/misc/rootql rootql

Open the ROOTQL project in Xcode and click on "Build" (make sure the Active
Build Configuration is set the "Release"). Copy the resulting
plugin from build/Release to the desired QuickLook directory.

Cheers, Fons.
