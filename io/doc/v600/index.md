## I/O Libraries

### TFileMerger

-   Added possibility to merge only a list of objects/folders from the
    input files, specified by name, \
    or to skip them from merging. This is fully integrated with the new
    PartialMerge(flags) schema. \
     Usage: \
    The names of the objects to be merged or skipped have to be
    specified using the interface:

``` {.cpp}
       TFileMerger::AddObjectNames(const char *names)
```

   This method can be called several times to add object names. Several
   names can be added with one call separated by single blancs (no
   blanc at the end). Directory names are accepted, applying the
   merging selection to all content. Two new options are being
   supported for partial merging:

``` {.cpp}
       TFileMerger::PartialMerge(flags | kOnlyListed)
```            

   This will merge only the objects in the files having the names in
   the specified list. If a folder is specified, it whole content will
   be merged

``` {.cpp}
        TFileMerger::PartialMerge(flags | kSkipListed)  
```           

   This will skip merging for the specified objects. If a folder is
   specified, its whole content will be skipped

   Important note:
   The kOnlyListed and kSkipListed flags have to be bitwise OR-ed 
   on top of the merging defaults: kAll | kIncremental (as in the example $ROOTSYS/tutorials/io/mergeSelective.C)

