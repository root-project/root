// @(#)root/winnt:$Id$
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

// USAGE: makemsi outputfile.msi -T filelist.txt
// will create a MSI file for files in filelist.txt

#if !defined(VERSION) || !defined(PRODUCT)
#  error "Define the CPP macros PRODUCT and VERSION!"
#endif

#include <rpc.h>
#include <stdio.h>
#include <list>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

using std::string;
using std::list;
using std::map;
using std::cerr;
using std::endl;
using std::ostream;

////////////////////////////////////////////////////////////////////////////////
// CLASS DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

class MSIDir;

class MSIDirEntry {
public:
   MSIDirEntry(const char* name, MSIDir* parent, bool dir);
   virtual ~MSIDirEntry() {}

   string GetLongName() const {return fLongName;}
   string GetShortName() const {return fShortName;}
   string GetPath() const {return fPath;}
   string GetId() const;
   MSIDir* GetParent() const {return fParent;}

   virtual void WriteRecurse(ostream& out, string indent) const = 0;
   ostream& WriteLongShort(ostream& out) const;

private:
   void SetShortName(bool dir);
   string GetMyId() const;

   string fLongName; // regular name
   string fShortName; // 8.3 name
   string fPath; // path incl parents
   MSIDir* fParent; // parent dir
};

////////////////////////////////////////////////////////////////////////////////

class MSIFile;

class MSIDir: public MSIDirEntry {
public:
   MSIDir(const char* name, MSIDir* parent=0): MSIDirEntry(name, parent, true) {}
   ~MSIDir() {if (!GetParent()) UpdateGuids();}

   void AddFile(const char* file);

   void Write(ostream& out) const;

private:
   void WriteRecurse(ostream& out, string indent) const;
   void WriteComponentsRecurse(ostream& out, string indent) const;
   const char* GetGuid() const {
      if (!fgGuids.size() && !fgNewGuids.size()) SetupGuids();
      map<string, string>::const_iterator iGuid = fgGuids.find(GetId());
      if (iGuid == fgGuids.end()) return CreateGuid();
      return iGuid->second.c_str();
   }
   const char* CreateGuid() const;
   static void SetupGuids();
   static void UpdateGuids();

   map<string, MSIDir*> fSubdirs;
   list<MSIFile*> fFiles;

   static map<string, string> fgGuids;
   static map<string, string> fgNewGuids; // guids created during this process
   static const char* fgGuidFileName; // location of the GUID file
};

////////////////////////////////////////////////////////////////////////////////

class MSIFile: public MSIDirEntry {
public:
   MSIFile(const char* name, MSIDir* parent): MSIDirEntry(name, parent, false) {}

   void WriteRecurse(ostream& out, string indent) const {
      out << indent << "<File Id=\"" << GetId() << "\" ";
      WriteLongShort(out) << "DiskId=\"1\"></File>" << std::endl;
   };
};



////////////////////////////////////////////////////////////////////////////////
// MSIDirEntry DEFINITIONS
////////////////////////////////////////////////////////////////////////////////

MSIDirEntry::MSIDirEntry(const char* name, MSIDir* parent, bool dir): fLongName(name), fParent(parent) 
{ 
   if (fParent) fPath = fParent->GetPath() + '/' + name;
   else fPath = ".";
   SetShortName(dir); 
}

void MSIDirEntry::SetShortName(bool dir) {
   WIN32_FIND_DATA findData;
   string filename(GetPath());
   HANDLE hFind = ::FindFirstFile(filename.c_str(), &findData);
   if (hFind == INVALID_HANDLE_VALUE) {
      cerr << "Cannot find " << filename << endl;
   } else {
      bool foundDir = (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) > 0;
      if (foundDir == !dir)
         cerr << filename << " is not what I expected it to be!" << endl;
      else
         fShortName = findData.cAlternateFileName;
   } 
   FindClose(hFind);
}

string MSIDirEntry::GetId() const { 
   string ret;
   if (fParent) ret = fParent->GetId() + "_";
   return ret + GetMyId();
}

string MSIDirEntry::GetMyId() const { 
   string ret(fLongName);
   std::replace(ret.begin(), ret.end(), '/', '_');
   std::replace(ret.begin(), ret.end(), '-', '_');
   std::replace(ret.begin(), ret.end(), '#', '_');
   std::replace(ret.begin(), ret.end(), '~', '_');
   std::replace(ret.begin(), ret.end(), '@', '_');
   return ret;
}

ostream& MSIDirEntry::WriteLongShort(ostream& out) const {
   if (!fShortName.empty()) 
      out << "LongName=\"" << fLongName <<"\" Name=\"" << fShortName << "\" ";
   else out << "Name=\"" << fLongName << "\" ";
   return out;
}



////////////////////////////////////////////////////////////////////////////////
// MSIDir DEFINITIONS
////////////////////////////////////////////////////////////////////////////////

map<string, string> MSIDir::fgGuids;
map<string, string> MSIDir::fgNewGuids;
const char* MSIDir::fgGuidFileName = 0; // set to e.g. "guids.txt" make GUIDs persistent

void MSIDir::AddFile(const char* file) {
   string subdir(file);
   string filename(file);

   string::size_type posSlash = subdir.find('/');
   if (posSlash != string::npos) {
      subdir.erase(posSlash, subdir.length());
      filename.erase(0, posSlash+1);
   } else subdir.erase();

   if (filename.empty()) {
      cerr << "Cannot add empty filename!" << endl;
      return;
   }
   if (subdir.empty()) fFiles.push_back(new MSIFile(filename.c_str(), this));
   else {
      if (!fSubdirs[subdir]) fSubdirs[subdir] = new MSIDir(subdir.c_str(), this);
      fSubdirs[subdir]->AddFile(filename.c_str());
   }
}

void MSIDir::Write(ostream& out) const {
   const DWORD bufsize = MAX_PATH;
   char pwd[bufsize];
   DWORD len = ::GetCurrentDirectory(bufsize, pwd);
   if (len > 0 && pwd[len - 1] == '\\') {
      pwd[len - 1] = 0;
   }

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
   out << "         <Environment Id=\"ROOTSYS\" Name=\"ROOTSYS\" Action=\"set\" Part=\"all\" System=\"yes\" Value=\"[INSTALLLOCATION]\" />" << std::endl;
   out << "         <Environment Id=\"PATHROOT\" Name=\"PATH\" Action=\"set\" Part=\"first\" System=\"yes\" Value=\"[INSTALLLOCATION]\\bin\" />" << std::endl;
   out << "         <Environment Id=\"PATHPY\" Name=\"PYTHONPATH\" Action=\"set\" Part=\"first\" System=\"yes\" Value=\"[INSTALLLOCATION]\\bin\" />" << std::endl;
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
   out << "         <Directory Id=\"INSTALLLOCATION\" Name=\"root\" FileSource=\"" << pwd << "\">" << std::endl;

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
   out << "   <Icon Id=\"root.exe\" SourceFile=\"" << pwd << "/bin/root.exe\" />" << std::endl;
   out << "   <UIRef Id=\"WixUI\" />" << std::endl;
   out << "</Product>" << std::endl;
   out << "</Wix>" << std::endl;
}

void MSIDir::WriteRecurse(ostream& out, string indent) const {
   // write to out recursively
   if (!fFiles.size() && !fSubdirs.size()) return;

   if (GetParent()) {
      // assume that Write takes care of the root dir.
      out << indent << "<Directory Id=\"" << GetId() << "\" ";
      WriteLongShort(out) << ">" << std::endl;
      indent+="   ";
   }

   if (fFiles.size()) {
      out << indent << "<Component Id=\"Component_" << GetId() << "\" Guid=\"" 
         << GetGuid() << "\">" << std::endl;
      indent+="   ";

      for (list<MSIFile*>::const_iterator iFile = fFiles.begin(); iFile != fFiles.end(); ++iFile) {
         (*iFile)->WriteRecurse(out, indent);
      }
      indent.erase(indent.length()-3, 3);
      out << indent << "</Component>" << std::endl;
   }
   for (map<string, MSIDir*>::const_iterator iSubdir = fSubdirs.begin(); iSubdir != fSubdirs.end(); ++iSubdir)
      iSubdir->second->WriteRecurse(out, indent);

   indent.erase(indent.length()-3, 3);
   if (GetParent()) {
      // assume that Write takes care of the root dir.
      out << indent << "</Directory>" << std::endl;
   }
}

void MSIDir::WriteComponentsRecurse(ostream& out, string indent) const {
   // write all components to out
   if (!fFiles.empty()) 
      out << indent << "<ComponentRef Id=\"Component_" << GetId() << "\" />" << std::endl;
   for (map<string, MSIDir*>::const_iterator iSubdir = fSubdirs.begin(); iSubdir != fSubdirs.end(); ++iSubdir)
      iSubdir->second->WriteComponentsRecurse(out, indent);
}


const char* MSIDir::CreateGuid() const {
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

void MSIDir::SetupGuids() {
   if (!fgGuidFileName) return;

   std::ifstream in(fgGuidFileName);
   std::string line;
   while (std::getline(in, line)) {
      std::istringstream sin(line);
      std::string id, guid;
      sin >> id >> guid;
      fgGuids[id] = guid;
   }
}

void MSIDir::UpdateGuids() {
   if (!fgNewGuids.size() || !fgGuidFileName) return;

   std::ofstream out(fgGuidFileName, std::ios_base::app);
   if (!out) {
      cerr << "ERROR: cannot write to GUID file " 
         << fgGuidFileName << "!" << endl;
      cerr << "ERROR: You should NOT use this MSI file, but re-generate with with accessible GUID file!"
         << endl;
      return;
   }
   for (map<string, string>::const_iterator iGuid = fgNewGuids.begin(); iGuid != fgNewGuids.end(); ++iGuid)
      out << iGuid->first << " " << iGuid->second << endl;
   std::cout << "WARNING: new GUIDs created; please cvs checkin " << fgGuidFileName << "!" << endl;
}



////////////////////////////////////////////////////////////////////////////////
// main()
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
   if (argc<4 || string(argv[2]) != "-T") {
      cerr << "USAGE: " << argv[0] << " <msifile> -T <inputlistfile>" << endl;
      return 1;
   }

   string outfile = argv[1];
   std::ofstream out(outfile.c_str());
   if (!out) {
      cerr << "Cannot open output file " << outfile << "!" << endl;
      return 2;
   }

   string infile = argv[3];
   std::ifstream in(infile.c_str());
   if (!in) {
      cerr << "Cannot open input file " << infile << "!" << endl;
      return 2;
   }

   MSIDir fileroot("ROOTSYS");
   string line;
   while (std::getline(in, line))
      fileroot.AddFile(line.c_str());

   fileroot.Write(out);
}
