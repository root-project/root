// $Id: VersionInfo.h,v 1.22 2013-03-01 14:24:48 avalassi Exp $
#ifndef CORALBASE_VERSIONINFO_H
#define CORALBASE_VERSIONINFO_H 1

// This switch is now hardcoded in the two branches of the code
// tagged as CORAL-preview and CORAL_2_3-patches (bug #89707).
//#define CORAL240 1 // CORAL 2.4.x
#undef CORAL240 // CORAL 2.3.x

#ifdef CORAL240
//---------------------------------------------------------------------------
// CORAL-preview (CORAL 2.4.x releases)
// Disable all extensions (do not allow explicit -D to enable them)
//---------------------------------------------------------------------------
#define CORAL_VERSIONINFO_RELEASE_MAJOR 2
#define CORAL_VERSIONINFO_RELEASE_MINOR 4
#define CORAL_VERSIONINFO_RELEASE_PATCH 0
#define CORAL240PR 1 // API switch IPropertyManager struct/class (bug #63198)
#define CORAL240PL 1 // API use new plugin loading as in clang (bug #92167)
#define CORAL240CO 1 // API fixes for Coverity (bugs #95355/8/9 and #95362)
#define CORAL240AL 1 // API extension for AttributeList::exists (task #20089)
#define CORAL240CL 1 // API fixes for clang (bug #100663)
#define CORAL240AS 1 // API ext. for AttributeList::specification (bug #100873)
#define CORAL240PM 1 // API fix for PropertyManager/MonitorObject (task #30840)
#define CORAL240MR 1 // API fix for MsgReporter (bug #53040)
#define CORAL240SO 1 // API fix for StringOps/StringList (bug #103240)
#define CORAL240CC 1 // API ext. for connection svc configuration (bug #100862)
#define CORAL240TS 1 // API remove TimeStamp::Epoch (bug #64016)
#define CORAL240EX 1 // API extension for exceptions (task #8688)
//---------------------------------------------------------------------------
#else
//---------------------------------------------------------------------------
// CORAL_2_3-patches (CORAL 2.3.x releases)
// Disable all extensions (do not allow explicit -D to enable them)
//---------------------------------------------------------------------------
#define CORAL_VERSIONINFO_RELEASE_MAJOR 2
#define CORAL_VERSIONINFO_RELEASE_MINOR 3
#define CORAL_VERSIONINFO_RELEASE_PATCH 29
#undef CORAL240PR // Do undef (do not leave the option to -D this explicitly)
#undef CORAL240PL // Do undef (do not leave the option to -D this explicitly)
#undef CORAL240CO // Do undef (do not leave the option to -D this explicitly)
#undef CORAL240AL // Do undef (do not leave the option to -D this explicitly)
#undef CORAL240CL // Do undef (do not leave the option to -D this explicitly)
#undef CORAL240AS // Do undef (do not leave the option to -D this explicitly)
#undef CORAL240PM // Do undef (do not leave the option to -D this explicitly)
#undef CORAL240MR // Do undef (do not leave the option to -D this explicitly)
#undef CORAL240SO // Do undef (do not leave the option to -D this explicitly)
#undef CORAL240CC // Do undef (do not leave the option to -D this explicitly)
#undef CORAL240TS // Do undef (do not leave the option to -D this explicitly)
#undef CORAL240EX // Do undef (do not leave the option to -D this explicitly)
//---------------------------------------------------------------------------
#endif

// CORAL_VERSIONINFO_RELEASE[_x] are #defined as of CORAL 2.3.13 (task #17431)
#define CORAL_VERSIONINFO_RELEASE CORAL_VERSIONINFO_RELEASE_MAJOR.CORAL_VERSIONINFO_RELEASE_MINOR.CORAL_VERSIONINFO_RELEASE_PATCH

#endif // CORALBASE_VERSIONINFO_H
