#include <CoreFoundation/CoreFoundation.h>
#include <CoreServices/CoreServices.h>
#include <QuickLook/QuickLook.h>
#include <libgen.h>

#import <Cocoa/Cocoa.h>

#include "ReadFile.h"

/* -----------------------------------------------------------------------------
   Generate a preview for a ROOT file
   ----------------------------------------------------------------------------- */

OSStatus GeneratePreviewForURL(void *thisInterface, QLPreviewRequestRef preview, CFURLRef url, CFStringRef contentTypeUTI, CFDictionaryRef options)
{
   NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];

#ifdef DEBUG
   NSDate *startDate = [NSDate date];
#endif

	// Get the posix-style path for the thing we are quicklooking at
	NSString *fullPath = (NSString*)CFURLCopyFileSystemPath(url, kCFURLPOSIXPathStyle);

#ifdef DEBUG
   NSLog(@"GeneratePreviewForURL %@", fullPath);
#endif

   // Check for cancel
	if (QLPreviewRequestIsCancelled(preview)) {
		[pool release];
		return noErr;
	}

   // Set properties for the preview data
	NSMutableDictionary *props = [[[NSMutableDictionary alloc] init] autorelease];
   [props setObject: @"UTF-8" forKey: (NSString *)kQLPreviewPropertyTextEncodingNameKey];
   [props setObject: @"text/html" forKey: (NSString *)kQLPreviewPropertyMIMETypeKey];
	//[props setObject: [NSString stringWithFormat: @"Contents of %@", fullPath] forKey: (NSString *)kQLPreviewPropertyDisplayNameKey];
   [props setObject: [NSString stringWithFormat: @"Contents of %s", basename((char*)[fullPath UTF8String])] forKey: (NSString *)kQLPreviewPropertyDisplayNameKey];

	// Build the HTML
   NSMutableString *html = [[[NSMutableString alloc] init] autorelease];
   [html appendString: @"<html>"];
   [html appendString: @"<head><style type=\"text/css\">"];
   [html appendString: @"body, td, th, p, div { font-family: Arial, Helvetica, sans-serif; font-size: 12px }"];
   [html appendString: @"</style></head>"];
   [html appendString: @"<body bgcolor=white>"];

   // Read ROOT file and fill html
   if (ReadFile(fullPath, html, preview) == -1) {
      [pool release];
      return noErr;
   }

   [html appendString: @"</body></html>"];

#ifdef DEBUG
   NSLog(@"Scanned file %@ in %.3f sec",
         fullPath, -[startDate timeIntervalSinceNow]);
#endif

   // Check for cancel
	if (QLPreviewRequestIsCancelled(preview)) {
		[pool release];
		return noErr;
	}

   // Now let WebKit do its thing
   QLPreviewRequestSetDataRepresentation(preview, (CFDataRef)[html dataUsingEncoding: NSUTF8StringEncoding], kUTTypeHTML, (CFDictionaryRef)props);

   [pool release];
   return noErr;
}

void CancelPreviewGeneration(void* thisInterface, QLPreviewRequestRef preview)
{
   // implement only if supported
}
