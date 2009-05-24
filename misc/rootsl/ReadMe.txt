ROOTSL
======

This is a Spotlight plugin that allows ROOT files to be indexed by SL.
Once indexed SL can find ROOT files based on the names and titles of the
objects in the files.

Spotlight is available on MacOS X since version 10.4 (Tiger). To use SL
select the SL icon on the top right of the menubar and type in a search text.

Get the binary for the ROOTSL plugin from:

   ftp://root.cern.ch/root/ROOTSL.tgz

To install the plugin, after untarring the above file, just drag the bundle
ROOTSL.mdimporter to /Library/Spotlight (global, i.e. for all users on a
system) or to ~/Library/Spotlight (local, this user only) directory.
You may need to create that folder if it doesn't already exist.

To build from source, get it from svn using:

   svn co http://root.cern.ch/svn/root/trunk/misc/rootsl rootsl

Open the ROOTSL project in Xcode and click on "Build" (make sure the Active
Build Configuration is set the "Release"). Copy the resulting
plugin from build/Release to the desired QuickLook directory.

Cheers, Fons.
