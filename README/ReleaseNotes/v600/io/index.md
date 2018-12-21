

## I/O Libraries

### I/O Behavior change.

#### Classes with custom streamer

Classes for which a Streamer function was externally provided are no longer
split; they were split in v5 if the dictionary was generated via rootcint but
were not split if the dictionary was generated via genreflex.

Classes with a custom Streamer function and classes with an older, non StreamerInfo
based automatic streamer are also no longer split.

To force the splitting of those classes, thus by-passing the custom Streamer,
when storing the object in a TTree and in a collection object, use:


``` {.cpp}
       TClass::GetClass(classname)->SetCanSplit(true);
```

### I/O Schema Checksum

The algorithm used to calculate a single number giving an indication on whether
the schema layout has changed (i.e. if two StreamerInfo are equivalent) have
been update to

- Use the normalized name for the types (i.e. two different spelling of the same
name will lead to the same checksum)
- Take into account the base classes' checksum in the derived class checksum;
this is necessary to properly support base classes during memberwise streaming.

The algorithm that checks whether two StreamerInfo are equal even-though their
checksum is different has been significantly enhanced in particular to also
check the base classes.

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

