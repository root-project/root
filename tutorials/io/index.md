\addtogroup tutorial_io
@{
## Table of contents
- [RNTuple](\ref tutorial_ntuple)
- [TTree](\ref tutorial_tree)
- [ROOT File](\ref file)
- [Merging multiple ROOT files](\ref merging)
- [FITS files](\ref tutorial_FITS)
- [SQL](\ref tutorial_sql)
- [XML](\ref tutorial_xml)
- [Various I/O related topics](\ref miscl)

\anchor file
## ROOT File
These are the tutorials illustrating the ROOT File and details of its directory-like structure. 
| **Tutorial** | **Description** |
|--------------|-----------------|
| file.C | Illustration of the ROOT File description |
| fildir.C | Illustration of the ROOT File directory-like structure |
| importCode.C | Create a ROOT File with sub-directories.|
| readCode.C | Navigate inside a ROOT file with sub-directories and read the objects from each sub-directory.|
| dirs.C | Create a hierarchy of directories in a ROOT File.  |
| copyFiles.C  | Copy all objects (including directories) from a source file. | 
| loopdir.C, loopdir11.C  |  Loop over all objects of a ROOT file directory and print all the TH1 derived objects in Postscript. |

\anchor merging
## Merging ROOT Files
A few examples and learning material on how to merge ROOT files together. 
| **Tutorial** | **Description** |
|--------------|-----------------|
| hadd.C |  Macro to add histogram files. **NOTE**: This macro is kept for didactical purposes only: use instead the executable $ROOTSYS/bin/hadd.|
| mergeSelective.C | Merge only part of the content of a set of files.|
| testMergeCont.C | Merge containers.|

\anchor miscl
## Various I/O related topics
The following tutorials illustrate various useful ROOT I/O features. 
| **Tutorial** || **Description** |
|------|--------|-----------------|
| double32.C |  | Details of the Double32_t data type - what is its precision and how to use it. |
| float16.C |  | Details of the Float16_t data type - what is its precision and how to use it. |
| testTMPIFile.C |  |  Usage of TMPIFile to simulate event reconstruction and merging them in parallel.|
|  | tcontext_context_manager.py |  Usage of the TContext class as a Python context manager. |
| |  tfile_context_manager.py |  Usage of TFile class as a Python context manager.|
@}


\defgroup tutorial_ntuple RNTuple tutorials
\ingroup tutorial_io
\brief Various examples demonstrating ROOT's RNTuple columnar I/O subsystem.

\defgroup tutorial_tree TTree tutorials
\ingroup tutorial_io
\brief Example code which illustrates how to use ROOT trees and ntuples.

\defgroup tutorial_FITS FITS files interface tutorials
\ingroup tutorial_io
\brief Examples showing the FITS file interface.

\defgroup tutorial_sql SQL tutorials
\ingroup tutorial_io
\brief Examples showing the SQL classes.

\defgroup tutorial_xml XML tutorials
\ingroup tutorial_io
\brief XML examples.