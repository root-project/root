#include <CoreFoundation/CoreFoundation.h>
#include <CoreServices/CoreServices.h>

#import <Foundation/Foundation.h>

#include "ReadFile.h"

/* -----------------------------------------------------------------------------
    Get metadata attributes from file

   This function's job is to extract useful information your file format supports
   and return it as a dictionary
   ----------------------------------------------------------------------------- */

Boolean GetMetadataForFile(void *thisInterface,
			   CFMutableDictionaryRef attributes,
			   CFStringRef contentTypeUTI,
			   CFStringRef pathToFile)
{
   // Pull any available metadata from the file at the specified path
   // Return the attribute keys and attribute values in the dict
   // Return TRUE if successful, FALSE if there was no data provided

   NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

   NSMutableSet *nameSet = [[[NSMutableSet alloc] init] autorelease];
   NSMutableSet *titleSet = [[[NSMutableSet alloc] init] autorelease];

   if (ReadFile((NSString*)pathToFile, nameSet, titleSet) == -1) {
      [pool release];
      return FALSE;
   }

   if ([nameSet count]) {
      NSString *names = [[nameSet allObjects] componentsJoinedByString: @"\n"];
      [(NSMutableDictionary *)attributes setObject: names
                                            forKey: @"ch_cern_root_data_objectName"];
   }
   if ([titleSet count]) {
      NSString *titles = [[titleSet allObjects] componentsJoinedByString: @"\n"];
      [(NSMutableDictionary *)attributes setObject: titles
                                            forKey: @"ch_cern_root_data_objectTitle"];
   }

   // memory management
   [pool release];

   return TRUE;
}
