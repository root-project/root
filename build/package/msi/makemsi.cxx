// @(#)root/winnt:$Name:  $:$Id: TWinNTSystem.cxx,v 1.136 2006/04/27 15:07:57 brun Exp $
// Author: Axel Naumann 2006-05-09

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// WiX MSI Installer Package Utility
// Creates a WiX source file

#include <windows.h>
#include <rpc.h>
#include <stdio.h>
#include <io.h>
#include <list>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

#ifndef FILEROOT
#  define FILEROOT ""
#endif

#ifndef DEFAULTFILTER
#  define DEFAULTFILTER "*.*"
#endif

#if !defined(VERSION) || !defined(PRODUCT)
#  error "Define the CPP macros PRODUCT and VERSION!"
#endif

class Subdir {
public:
   typedef std::pair<std::string, std::string> LongShortName;
   typedef std::list<Subdir*> Subdirs;
   typedef Subdirs::const_iterator CISubdirs;
   typedef std::list<LongShortName> Files;
   typedef Files::const_iterator CIFiles;
   typedef std::vector<std::string> Slices;
   typedef Slices::const_iterator CISlices;

   Subdir() {}
   Subdir(LongShortName relpath, Subdir* parent=0, const char* fileFilter = 0, bool recurse = false): 
      fName(relpath), fParent(parent) {
      if (!parent && !fgGuids.size()) SetupGuids(relpath.first.c_str());

      if (fParent) {
         std::cout << "processing ";
         Subdir* parent = this;
         std::string path = relpath.first;
         while (parent = parent->GetParent()) 
            path = parent->GetName().first + "/" + path;;
         std::cout << path << "..." << std::endl;
      }

      std::replace(fName.first.begin(), fName.first.end(), '\\', '/');
      if (!fName.first.empty() && fName.first[fName.first.length()-1] == '/') 
         fName.first.erase(fName.first.length()-1,1);

      std::replace(fName.second.begin(), fName.second.end(), '\\', '/');
      if (!fName.second.empty() && fName.second[fName.second.length()-1] == '/') 
         fName.second.erase(fName.second.length()-1,1);

      if (fileFilter) {
         AddFiles(fileFilter);
         if (recurse) AddSubdirs(DEFAULTFILTER, fileFilter);
      }
   }

   ~Subdir() { if (IsValid() && !fParent) UpdateGuids(fName.first.c_str()); }

   bool IsValid() const { return !fName.first.empty(); }
   LongShortName GetName() const { return fParent ? fName : LongShortName(fName.second,fName.second); }
   const LongShortName& GetPath() const { return fName; }
   Subdir* GetParent() const { return fParent; }

   std::string GetSubId(const char* file = 0) const {
      std::string ret(GetName().first);
      if (file) ret += std::string("_") + file;
      std::replace(ret.begin(), ret.end(), '/', '_');
      std::replace(ret.begin(), ret.end(), '-', '_');
      return ret; 
   }

   std::string GetId(const char* file = 0) const { 
      return fParent ? fParent->GetId() + "_" + GetSubId(file) : GetSubId(file);
   }

   std::string GetFullPath() const { 
      return fParent ? fParent->GetFullPath() + "/" + fName.first : "."; }

   const char* GetGuid() const {
      CIMapGUIDs iGuid = fgGuids.find(GetId());
      if (iGuid == fgGuids.end()) return CreateGuid();
      return iGuid->second.c_str();
   }

   void Write(std::ostream& out) const;

   int AddFiles(const char* filter = DEFAULTFILTER);
   const Files &GetFiles() const { return fFiles; }

   Subdir& AddSubdir(const char* relpath, const char* filefilter = DEFAULTFILTER);
   int AddSubdirs(const char* dirfilter = DEFAULTFILTER, const char* filefilter = DEFAULTFILTER);
   const Subdirs &GetSubdirs() const { return fSubdirs; }

   static Slices Slice(const std::string& in, const char* delim=";", bool requireAll = true);
   static void SetupGuids(const char* root);
   static void UpdateGuids(const char* root);

private:
   std::ostream& WriteLongShort(std::ostream& out, const LongShortName& what) const {
      if (!what.second.empty()) 
         out << "LongName=\"" << what.first <<"\" Name=\"" << what.second << "\" ";
      else out << "Name=\"" << what.first << "\" ";
      return out;
   }
   const char* CreateGuid() const {
      UUID uuid;
      ::UuidCreate(&uuid);
      unsigned char* str = 0;
      ::UuidToString(&uuid, &str);
      std::string id = GetId();
      const std::string& ret = fgGuids[id] = (char*)str;
      fgNewGuids[id] = ret;
      RpcStringFree(&str);
      return ret.c_str();
   }
   void WriteRecurse(std::ostream& out, std::string& indent) const;
   void WriteComponentsRecurse(std::ostream& out, std::string& indent) const;

   typedef std::map<std::string, std::string> MapGUIDs;
   typedef std::map<std::string, std::string>::const_iterator CIMapGUIDs;

   Subdir* fParent; // parent dir
   LongShortName fName; // name
   Subdirs fSubdirs; // subdirs
   Files fFiles; // files in dir
   static MapGUIDs fgGuids;
   static MapGUIDs fgNewGuids; // guids created during this process
   static const char* fgGuidFileName; // location of the GUID file
};

Subdir::MapGUIDs Subdir::fgGuids;
Subdir::MapGUIDs Subdir::fgNewGuids;
const char* Subdir::fgGuidFileName = "build/package/msi/guids.txt";

void Subdir::SetupGuids(const char* root) {
   std::ifstream in((std::string(root)+ "/" + fgGuidFileName).c_str());
   std::string line;
   while (std::getline(in, line)) {
      std::istringstream sin(line);
      std::string id, guid;
      sin >> id >> guid;
      fgGuids[id] = guid;
   }
}

void Subdir::UpdateGuids(const char* root) {
   if (!fgNewGuids.size()) return;

   std::ofstream out((std::string(root)+"/" + fgGuidFileName).c_str(), std::ios_base::app);
   if (!out) {
      std::cerr << "ERROR: cannot write to GUID file " 
         << root << "/" << fgGuidFileName << "!" << std::endl;
      return;
   }
   for (CIMapGUIDs iGuid = fgNewGuids.begin(); iGuid != fgNewGuids.end(); ++iGuid)
      out << iGuid->first << " " << iGuid->second << std::endl;
   std::cout << "WARNING: new GUIDs created; cvs checkin " << fgGuidFileName << "!" << std::endl;
}

void Subdir::Write(std::ostream& out) const {
   if (!fFiles.size() && !fSubdirs.size() || !IsValid()) return;

   out << "<?xml version=\"1.0\" encoding=\"windows-1252\" ?>" << std::endl;
   out << "<Wix xmlns=\"http://schemas.microsoft.com/wix/2003/01/wi\">" << std::endl;
   out << "<Product Name=\"" << PRODUCT << "\" Id=\"{f570a3d8-bc0d-408e-bbe3-57e6deee5aaa}\" Language=\"1033\" Codepage=\"1252\"" << std::endl;
   out << "   Version=\"" << VERSION << "\" Manufacturer=\"ROOT Team\">" << std::endl;
   out << "   <Package Id=\"???????\?-???\?-???\?-???\?-????????????\" Keywords=\"Installer\" Description=\"ROOT Installer\"" << std::endl;
   out << "        Comments=\"ROOT Windows Installer, see http://root.cern.ch\" Manufacturer=\"CERN\" InstallerVersion=\"100\"" << std::endl;
   out << "        Languages=\"1033\" Compressed=\"yes\" SummaryCodepage=\"1252\" />" << std::endl;
   out << "   <Media Id=\"1\" Cabinet=\"ROOT.cab\" EmbedCab=\"yes\" />" << std::endl;
   out << "   <Directory Id=\"TARGETDIR\" Name=\"SourceDir\">" << std::endl;
   out << "      <Component Id=\"EnvVars\" Guid=\"{0E85B756-F20C-4213-A292-4AD80A5FE21A}\">" << std::endl;
   out << "         <Environment Id=\"ROOTSYS\" Name=\"ROOTSYS\" Action=\"set\" Part=\"all\" Value=\"[INSTALLLOCATION]\" />" << std::endl;
   out << "         <Environment Id=\"PATHROOT\" Name=\"PATH\" Action=\"set\" Part=\"first\" Value=\"[INSTALLLOCATION]bin\" />" << std::endl;
   out << "         <Environment Id=\"PATHPY\" Name=\"PYTHONPATH\" Action=\"set\" Part=\"first\" Value=\"[INSTALLLOCATION]pybin\" />" << std::endl;
   out << "      </Component>" << std::endl;
   out << "      <Directory Id=\"DesktopFolder\">" << std::endl;
   out << "         <Component Id=\"DesktopIcon\" Guid=\"{BF68C3D3-D9AC-488d-A73F-1C732466DBF7}\">" << std::endl;
   out << "            <Shortcut Id=\"DesktopIcon\" Directory=\"DesktopFolder\"" << std::endl;
   out << "               Name=\"" << PRODUCT << "\" LongName=\"" << PRODUCT << " " << VERSION << "\" Target=\"[!ROOTSYS_bin_root.exe]\" " << std::endl;
   out << "               WorkingDirectory=\"INSTALLLOCATION\" Icon=\"root.exe\" IconIndex=\"0\" />" << std::endl;
   out << "         </Component>" << std::endl;
   out << "      </Directory>" << std::endl;
   out << "      <Directory Id=\"ProgramMenuFolder\">" << std::endl;
   out << "         <Directory Id=\"PM_ROOT\" Name=\"ROOT\">" << std::endl;
   out << "            <Component Id=\"StartMenuIcon\" Guid=\"{801AF243-0D4F-4f26-AF98-60DFB704969E}\">" << std::endl;
   out << "               <Shortcut Id=\"StartMenuIcon\" Directory=\"PM_ROOT\"" << std::endl;
   out << "                 Name=\"" << PRODUCT << "\" LongName=\"" << PRODUCT << " " << VERSION << "\" Target=\"[!ROOTSYS_bin_root.exe]\" " << std::endl;
   out << "                 WorkingDirectory=\"INSTALLLOCATION\" Icon=\"root.exe\" IconIndex=\"0\" />" << std::endl;
   out << "            </Component>" << std::endl;
   out << "         </Directory>" << std::endl;
   out << "      </Directory>" << std::endl;
   out << "      <Directory Id=\"WindowsVolume\" Name=\"WinVol\">" << std::endl;
   out << "         <Directory Id=\"INSTALLLOCATION\" Name=\"root\" FileSource=\"" << fName.first << "\">" << std::endl;

   WriteRecurse(out, std::string("             "));

   out << "         </Directory>" << std::endl;
   out << "      </Directory>" << std::endl;
   out << "   </Directory>" << std::endl;
   out << "   <Feature Id=\"Base\" Title=\"Core modules\" Level=\"1\" Description=\"Core ROOT files\" Absent=\"disallow\" "<< std::endl;
   out << "      AllowAdvertise =\"no\" ConfigurableDirectory=\"INSTALLLOCATION\" TypicalDefault=\"install\">" << std::endl;

   WriteComponentsRecurse(out, std::string("         "));

   out << "      <Feature Id=\"EnvVars\" Title=\"Environment variables\" Level=\"1\" Description=\"ROOT environment variables\" AllowAdvertise =\"no\">" << std::endl;
   out << "         <ComponentRef Id=\"EnvVars\" />" << std::endl;
   out << "      </Feature>" << std::endl;
   out << "   </Feature>" << std::endl;
   out << "   <Feature Id=\"Shortcuts\" Title=\"Shortcuts\" Level=\"1\" Description=\"ROOT shortcuts\" AllowAdvertise =\"no\">" << std::endl;
   out << "      <Feature Id=\"DesktopIcon\" Title=\"Desktop Icon\" Level=\"1\" Description=\"ROOT desktop icon\">" << std::endl;
   out << "         <ComponentRef Id=\"DesktopIcon\" />" << std::endl;
   out << "      </Feature>" << std::endl;
   out << "      <Feature Id=\"StartMenuIcon\" Title=\"Startmenu Icon\" Level=\"1\" Description=\"ROOT start menu entry\">" << std::endl;
   out << "         <ComponentRef Id=\"StartMenuIcon\" />" << std::endl;
   out << "      </Feature>" << std::endl;
   out << "   </Feature>" << std::endl;
   out << "   <Property Id=\"INSTALLLOCATION\">" << std::endl;
   out << "      <RegistrySearch Id=\"RegInstallLocation\" Type=\"raw\" " << std::endl;
   out << "         Root=\"HKLM\" Key=\"Software\\CERN\\ROOT\" Name=\"InstallDir\" />" << std::endl;
   out << "   </Property>" << std::endl;
   out << "   <Icon Id=\"root.exe\" SourceFile=\"" << fName.first << "/bin/root.exe\" />" << std::endl;
   out << "   <UIRef Id=\"WixUI\" />" << std::endl;
   out << "</Product>" << std::endl;
   out << "</Wix>" << std::endl;
}

void Subdir::WriteRecurse(std::ostream& out, std::string& indent) const {
   // write to out recursively
   if (!fFiles.size() && !fSubdirs.size() || !IsValid()) return;

   if (fParent) {
      // assume that Write takes care of the root dir.
      out << indent << "<Directory Id=\"" << GetId() << "\" ";
      WriteLongShort(out, GetPath()) << ">" << std::endl;
      indent+="   ";
   }

   if (fFiles.size()) {
      out << indent << "<Component Id=\"Component_" << GetId() << "\" Guid=\"" 
         << GetGuid() << "\">" << std::endl;
      indent+="   ";

      for (CIFiles iFile = GetFiles().begin(); iFile != GetFiles().end(); ++iFile) {
         out << indent << "<File Id=\"" << GetId(iFile->first.c_str()) << "\" ";
         WriteLongShort(out, *iFile) << "DiskId=\"1\"></File>" << std::endl;
      }
      indent.erase(indent.length()-3, 3);
      out << indent << "</Component>" << std::endl;
   }
   for (CISubdirs iSubdir = fSubdirs.begin(); iSubdir != fSubdirs.end(); ++iSubdir)
      (*iSubdir)->WriteRecurse(out, indent);

   indent.erase(indent.length()-3, 3);
   if (fParent) {
      // assume that Write takes care of the root dir.
      out << indent << "</Directory>" << std::endl;
   }
}

void Subdir::WriteComponentsRecurse(std::ostream& out, std::string& indent) const {
   // write all components to out
   if (!IsValid()) return;
   if (!fFiles.empty()) 
      out << indent << "<ComponentRef Id=\"Component_" << GetId() << "\" />" << std::endl;
   for (CISubdirs iSubdir = fSubdirs.begin(); iSubdir != fSubdirs.end(); ++iSubdir)
      (*iSubdir)->WriteComponentsRecurse(out, indent);
}

Subdir::Slices Subdir::Slice(const std::string& in, const char* delim, bool requireAll) {
// Return the parts of in delimited by delim. 
// If requireAll, the string delim is used as delimiter, otherwise
// any character of delim found in in will serve as delimiter.
   typedef std::string::size_type PosDelim;
   typedef std::list<PosDelim> PosDelimList;
   typedef PosDelimList::const_iterator CIPosDelimList;
   PosDelimList listPosDelim;
   PosDelim posDelim = 0;
   while (requireAll && std::string::npos != (posDelim = in.find(delim, posDelim))
      || !requireAll && std::string::npos != (posDelim = in.find_first_of(delim, posDelim))) {
      listPosDelim.push_back(posDelim);
      ++posDelim;
   }
   listPosDelim.push_back(in.length());

   std::vector<std::string> ret(listPosDelim.size());
   size_t idx = 0;
   CIPosDelimList nextPosDelim = listPosDelim.begin();
   ++nextPosDelim;
   ret[idx++] = in.substr(0, *listPosDelim.begin());
   for (CIPosDelimList iPosDelim = listPosDelim.begin(); 
      nextPosDelim != listPosDelim.end(); ++iPosDelim) {
      PosDelim len = *nextPosDelim - *iPosDelim;
      ret[idx++] = in.substr(*iPosDelim + 1, len - 1);
      ++nextPosDelim;
   }
   return ret;
}

int Subdir::AddFiles(const char* filter) {
   if (!IsValid()) return 0;

   size_t oldSize = fFiles.size();
   Slices filters = Slice(std::string(filter));
   for (CISlices iFilter = filters.begin(); iFilter != filters.end(); ++iFilter) {
      WIN32_FIND_DATA findData;
      std::string filename(GetFullPath() + "/" + *iFilter);
      HANDLE hFind = ::FindFirstFile((LPCTSTR)filename.c_str(), &findData);
      if (hFind == INVALID_HANDLE_VALUE) continue;
      do {
         if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
         fFiles.push_back(
            std::make_pair(
               std::string((const char*)findData.cFileName), 
               std::string((const char*)findData.cAlternateFileName)));
      }  while (FindNextFile(hFind, &findData));
      FindClose(hFind);
   }

   return (int)(fFiles.size() - oldSize);
}

Subdir &Subdir::AddSubdir(const char* relpath, const char* filefilter) {
   static Subdir invalid;
   if (!IsValid()) return invalid;

   WIN32_FIND_DATA findData;
   std::string filename(GetFullPath() + "/" + relpath);
   HANDLE hFind = FindFirstFile((LPCTSTR)filename.c_str(), &findData);
   if (hFind == INVALID_HANDLE_VALUE || !(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) 
      return invalid;

   fSubdirs.push_back(
      new Subdir(
         std::make_pair(
            std::string((const char*)findData.cFileName), 
            std::string((const char*)findData.cAlternateFileName)),
         this, filefilter));
   FindClose(hFind);
   return *fSubdirs.back();
}

int Subdir::AddSubdirs(const char* dirfilter, const char* filefilter) {
   if (!IsValid()) return 0;

   size_t oldSize = fSubdirs.size();
   Slices filters = Slice(std::string(dirfilter));
   for (CISlices iFilter = filters.begin(); iFilter != filters.end(); ++iFilter) {
      WIN32_FIND_DATA findData;
      std::string filename(GetFullPath() + "/" + *iFilter);
      HANDLE hFind = FindFirstFile((LPCTSTR)filename.c_str(), &findData);
      if (hFind == INVALID_HANDLE_VALUE) continue;
      do {
         if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
         if (findData.cFileName[0] == '.') continue; // ., .., hidden
         if (!strcmp(findData.cFileName, "CVS")) continue;
         fSubdirs.push_back(
            new Subdir(
               std::make_pair(
                  std::string((const char*)findData.cFileName), 
                  std::string((const char*)findData.cAlternateFileName)),
               this, filefilter, dirfilter && !strcmp(dirfilter, DEFAULTFILTER)));
      }  while (FindNextFile(hFind, &findData));
      FindClose(hFind);
   }

   return (int)(fSubdirs.size() - oldSize);
}

int CreateXMSForROOT(const char* fileroot, const char* outpath) {
   // that's the workhorse for ROOT.

   Subdir root(Subdir::LongShortName(fileroot, "ROOTSYS"), 0, "LICENSE");
   root.AddSubdir("bin", "*.dll;*.exe;*.pdb;*.bat;*.py;*.pyc;*.pyo;memprobe;root-config");
   root.AddSubdir("lib", "*.lib;*.exp;*.def")
       .AddSubdir("python", 0)
       .AddSubdir("genreflex", "*.py;*.pyc");

   root.AddSubdir("include").AddSubdirs();

   Subdir &cint = root.AddSubdir("cint", "MAKEINFO");
   Subdir &cintinc = cint.AddSubdir("include");
   cintinc.AddSubdirs("GL;sys;X11");
   Subdir &cintlib = cint.AddSubdir("lib");
   cintlib.AddSubdirs("gl;stdstrct;qt;vcstream;vc7strm;win32api;xlib");
   cintlib.AddSubdir("dll_stl","*.h;README.txt;setup*.*");
   cint.AddSubdir("stl");

   root.AddSubdir("icons", "*.png;*.xpm");
   root.AddSubdir("fonts", "*.ttf;LICENSE");
   root.AddSubdir("tutorials");
   root.AddSubdir("macros");
   root.AddSubdir("test").AddSubdirs();
   root.AddSubdir("man", 0).AddSubdir("man1");
   root.AddSubdir("etc").AddSubdirs("daemons;proof");
   root.AddSubdir("build", 0).AddSubdir("misc", "root.m4;root-help.el");
   root.AddSubdir("config", "Makefile.win32;Makefile.config");
   root.AddSubdir("README");

   std::cout << "Writing " << outpath << std::endl;
   std::ofstream out(outpath);
   root.Write(out);

   return 0;
}

int main(int argc, char *argv[]) {
   const DWORD bufsize = MAX_PATH;
   char buf[bufsize];
   std::string fileroot = FILEROOT;
   if (fileroot.empty()) {
      ::GetCurrentDirectory(bufsize, buf);
      fileroot = buf;
   }
   if (::_access(fileroot.c_str(), 04 /*r*/)) {
      std::cerr << "ERROR: Cannot access " << fileroot << std::endl;
      return 1;
   }

   std::string outpath = "./";
   if (argc>1) outpath = argv[1];
   if (::_access(outpath.c_str(), 06 /*rw*/)) {
      std::cerr << "ERROR: Cannot access output path " << outpath << std::endl;
      return 2;
   }
   if (outpath[outpath.length()-1]!='/' && outpath[outpath.length()-1]!='\\')
      outpath += "/";
   outpath += "ROOT.xms";
   return CreateXMSForROOT(fileroot.c_str(), outpath.c_str());
}
