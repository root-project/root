#include <cassert>

#include <Cocoa/Cocoa.h>

#include "CocoaConstants.h"
#include "MenuLoader.h"
#include "CocoaUtils.h"

namespace ROOT {
namespace MacOSX {
namespace Details {

//Fill ROOT's menu (standard Apple's menu at the top of desktop).

void PopulateApplicationMenu(NSMenu *submenu);
void PopulateWindowMenu(NSMenu *submenu);
void PopulateHelpMenu(NSMenu *submenu);

//Fill app's menu.

//______________________________________________________________________________
void PopulateMainMenu()
{
   const Util::AutoreleasePool pool;

   NSMenu * const mainMenu = [[NSMenu alloc] initWithTitle : @"NSMainMenu"];
   const Util::NSScopeGuard<NSMenu> mainMenuGuard(mainMenu);

   // The strings in the menu bar come from the submenu titles,
   // except for the application menu, whose title is ignored at runtime.
   NSMenuItem *menuItem = [mainMenu addItemWithTitle : @"Apple" action : nil keyEquivalent:@""];
   NSMenu *submenu = [[NSMenu alloc] initWithTitle : @"Apple"];
   const Util::NSScopeGuard<NSMenu> submenuGuard1(submenu);

   PopulateApplicationMenu(submenu);
   [mainMenu setSubmenu : submenu forItem : menuItem];

   menuItem = [mainMenu addItemWithTitle : @"Window" action : nil keyEquivalent : @""];
   submenu = [[NSMenu alloc] initWithTitle : NSLocalizedString(@"Window", @"The Window menu")];
   const Util::NSScopeGuard<NSMenu> submenuGuard2(submenu);
   PopulateWindowMenu(submenu);
   [mainMenu setSubmenu : submenu forItem : menuItem];
   [NSApp setWindowsMenu : submenu];

   menuItem = [mainMenu addItemWithTitle:@"Help" action:NULL keyEquivalent:@""];
   submenu = [[NSMenu alloc] initWithTitle:NSLocalizedString(@"Help", @"The Help menu")];
   const Util::NSScopeGuard<NSMenu> submenuGuard3(submenu);
   PopulateHelpMenu(submenu);
   [mainMenu setSubmenu : submenu forItem : menuItem];

   [NSApp setMainMenu : mainMenu];
   [NSMenu setMenuBarVisible : YES];
}

//______________________________________________________________________________
void PopulateApplicationMenu(NSMenu *aMenu)
{
   assert(aMenu != nil && "PopulateApplicationMenu, aMenu parameter is nil");

   NSString * const applicationName = @"root";

   NSMenuItem *menuItem = [aMenu addItemWithTitle : [NSString stringWithFormat : @"%@ %@",
                                          NSLocalizedString(@"About", nil), applicationName]
                           action : @selector(orderFrontStandardAboutPanel:) keyEquivalent : @""];
   [menuItem setTarget : NSApp];
   [aMenu addItem : [NSMenuItem separatorItem]];

   menuItem = [aMenu addItemWithTitle : [NSString stringWithFormat : @"%@ %@",
               NSLocalizedString(@"Hide", nil), applicationName] action : @selector(hide:) keyEquivalent : @"h"];
   [menuItem setTarget : NSApp];

   menuItem = [aMenu addItemWithTitle : NSLocalizedString(@"Hide Others", nil)
               action : @selector(hideOtherApplications:) keyEquivalent : @"h"];
   [menuItem setKeyEquivalentModifierMask : Details::kCommandKeyMask | Details::kAlternateKeyMask];
   [menuItem setTarget : NSApp];

   menuItem = [aMenu addItemWithTitle : NSLocalizedString(@"Show All", nil)
               action : @selector(unhideAllApplications:) keyEquivalent : @""];
   [menuItem setTarget : NSApp];

   [aMenu addItem : [NSMenuItem separatorItem]];
   menuItem = [aMenu addItemWithTitle : [NSString stringWithFormat : @"%@ %@",
               NSLocalizedString(@"Quit", nil), applicationName] action : @selector(terminate:) keyEquivalent : @"q"];
   [menuItem setTarget : NSApp];
}

//______________________________________________________________________________
void PopulateWindowMenu(NSMenu *aMenu)
{
   assert(aMenu != nil && "PopulateWindowMenu, aMenu parameter is nil");

   NSMenuItem *menuItem = [aMenu addItemWithTitle : NSLocalizedString(@"Minimize", nil)
                           action : @selector(performMinimize:) keyEquivalent : @"m"];
   menuItem = [aMenu addItemWithTitle : NSLocalizedString(@"Zoom", nil)
               action : @selector(performZoom:) keyEquivalent : @""];
   [aMenu addItem : [NSMenuItem separatorItem]];
   menuItem = [aMenu addItemWithTitle : NSLocalizedString(@"Bring All to Front", nil)
               action : @selector(arrangeInFront:) keyEquivalent : @""];
}

//______________________________________________________________________________
void PopulateHelpMenu(NSMenu *aMenu)
{
   NSMenuItem * const menuItem = [aMenu addItemWithTitle : [NSString stringWithFormat : @"%@ %@", @"root",
                                  NSLocalizedString(@"Help", nil)] action : @selector(showHelp:) keyEquivalent : @"?"];
   [menuItem setTarget : NSApp];
}

}//Detail
}//MacOSX
}//ROOT
