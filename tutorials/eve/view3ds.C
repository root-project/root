// @(#)root/eve:$Id$
// Author: Bertrand Bellenot

// Loading and display of basic 3DS models.

#include "TCanvas.h"
#include "TStyle.h"
#include "TFile.h"
#include "TStopwatch.h"
#include "TError.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

class TEveTriangleSet;

TEveTriangleSet *ts[2048];

// Believe3D Model file defines
#define MAGICNUMBER 0xB3D0

// types of 3DS Chunks
#define CHUNKMAIN                   0x4D4D
#define CHUNKMAINVERSION            0x0002
#define CHUNK3D                     0x3D3D
#define CHUNK3DVERSION              0x3D3E
#define CHUNK3DOBJECT               0x4000
#define CHUNK3DOBJECTMESH           0x4100
#define CHUNK3DOBJECTMESHVERTICES   0x4110
#define CHUNK3DOBJECTMESHFACES      0x4120
#define CHUNK3DOBJECTMESHMATGROUP   0x4130
#define CHUNK3DOBJECTMESHMAPPING    0x4140

#define CHUNK3DMATERIAL             0xAFFF
// Sub defines of MATERIAL
#define MATNAME                     0xA000
#define MATDIFFUSE                  0xA020
#define MATSPECULAR                 0xA030
#define MATTRANSPARENCY             0xA050

#define COLOR_F                     0x0010
#define COLOR_24                    0x0011
#define LIN_COLOR_24                0x0012
#define LIN_COLOR_F                 0x0013
#define INT_PERCENTAGE              0x0030
#define FLOAT_PERCENTAGE            0x0031

//////////////////////////////////////
//The tMaterialInfo Struct
//////////////////////////////////////
class Material {
public:
   char     name[256];
   UChar_t  color[3];
   UShort_t transparency;

   Material() {
      sprintf(name, "");
      color[0] = color[1] = color[2] = 0;
      transparency = 0;
   }
   ~Material() { }
};


// Chunk structure
typedef struct _Chunk {
   UShort_t idnum;
   UInt_t   offset, len, endoffset;
} Chunk;

// vertex structure
typedef struct _Vertex {
   Float_t x, y, z;
   Float_t u, v;
} Vertex;

// face structure
typedef struct _Face {
   UInt_t  v1, v2, v3;
} Face;

// model structure
class Model {
public:
   char     name[256];
   char     matname[256];
   Vertex   *vlist;
   Face     *flist;
   UInt_t   numverts, numfaces;

   Model() {
      sprintf(name,"");
      sprintf(matname,"");
      vlist = 0;
      flist = 0;
      numverts = numfaces = 0;
   }
   ~Model() {
      if (vlist != 0) delete [] vlist;
      if (flist != 0) delete [] flist;
   }
};

// chunk reading routines
Int_t ReadChunk(FILE*, Chunk*);

// data reading routines
Int_t ReadMainChunk(FILE*);
Int_t Read3DChunk(FILE*, UInt_t);
Int_t ReadObjectChunk(FILE*, UInt_t);
Int_t ReadMeshChunk(FILE*, UInt_t, char*);
Int_t ReadVerticesChunk(FILE*);
Int_t ReadFacesChunk(FILE*);
Int_t ReadMappingChunk(FILE*);
Int_t ReadASCIIZ(FILE*, char*);
Int_t ReadMaterialChunk(FILE *, UInt_t);
Int_t ReadColor(FILE *, UInt_t);
Int_t ReadTransparency(FILE *, UInt_t);
Int_t ReadObjectMaterial(FILE *);
Int_t ConvertModel();

// global variables
Int_t nummodels = 0;
Model model;

Int_t nummaterials = 0;
Material *material[1024];

//______________________________________________________________________________
Int_t Read3DSFile(const char *fname)
{
   // main function

   FILE *infile;

   infile = fopen(fname, "rb");
   if (infile == 0) {
      printf("Error : Input File Could Not Be Opened!\n");
      return -1;
   }
   UShort_t magic = MAGICNUMBER;
   if (ReadMainChunk(infile) != 0) {
      printf("Error : Input File Could Not Be Read!\n");
   }
   fclose(infile);
   return 0;
}

//______________________________________________________________________________
Int_t ReadChunk(FILE *f, Chunk *c)
{
   // reads a chunk from an opened file

   if (feof(f)) return(-1);
   c->idnum = 0;
   c->offset = c->len = 0;
   c->offset = (UInt_t) ftell(f);
   fread(&c->idnum, sizeof(UShort_t), 1, f);
   fread(&c->len, sizeof(UInt_t), 1, f);
   c->endoffset = c->offset + c->len;
   return(0);
}

//______________________________________________________________________________
Int_t ReadMainChunk(FILE *f)
{
   // handles the main body of the 3DS file

   Chunk chunk;

   ReadChunk(f, &chunk);
   if (chunk.idnum != CHUNKMAIN) return(-1);
   while ((ReadChunk(f, &chunk) == 0) && (!feof(f))) {
      if (chunk.idnum == CHUNK3D) {
         Read3DChunk(f, chunk.endoffset);
      }
      else {
         //printf("Debug : Unknown Chunk [Main Chunk] [0x%x]\n", chunk.idnum);
         fseek(f, chunk.offset + chunk.len, SEEK_SET);
      }
   }
   return 0;
}

//______________________________________________________________________________
Int_t Read3DChunk(FILE *f, UInt_t len)
{
   // reads the 3D Edit Chunk

   Chunk chunk;

   while ((ReadChunk(f, &chunk) == 0) && (!feof(f))) {
      if (chunk.idnum == CHUNK3DOBJECT) {
         ReadObjectChunk(f, chunk.endoffset);
         fseek(f, chunk.endoffset, SEEK_SET);
      }
      else if (chunk.idnum == CHUNK3DMATERIAL) {
         ReadMaterialChunk(f, chunk.endoffset);
         fseek(f, chunk.endoffset, SEEK_SET);
      }
      else {
         if (chunk.endoffset < len) {
            //printf("Debug : Unknown Chunk [3D Chunk] [0x%x]\n", chunk.idnum);
            fseek(f, chunk.endoffset, SEEK_SET);
         }
         else {
            break;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
Int_t ReadMaterialChunk(FILE *f, UInt_t len)
{
   // reads the Material sub-chunk of the 3D Edit Chunk

   Chunk chunk;
   char name[256];
   material[nummaterials] = new Material();
   while ((ReadChunk(f, &chunk) == 0) && (!feof(f))) {
      if (chunk.idnum == MATNAME) {
         ReadASCIIZ(f, name);
         strcpy(material[nummaterials]->name, name);
         fseek(f, chunk.endoffset, SEEK_SET);
      }
      else if (chunk.idnum == MATDIFFUSE) {
         ReadColor(f, chunk.endoffset);
         fseek(f, chunk.endoffset, SEEK_SET);
      }
      else if (chunk.idnum == MATTRANSPARENCY) {
         ReadTransparency(f, chunk.endoffset);
         fseek(f, chunk.endoffset, SEEK_SET);
      }
      else {
         if (chunk.endoffset < len) {
            //printf("Debug : Unknown Chunk [Object Chunk] [0x%x]\n", chunk.idnum);
            fseek(f, chunk.endoffset, SEEK_SET);
         }
         else {
            break;
         }
      }
   }
   nummaterials++;
   return 0;
}

//______________________________________________________________________________
Int_t ReadColor(FILE *f, UInt_t len)
{
   // reads the Color property of the Material Chunk

   Chunk chunk;
   float fr, fg, fb;
   while ((ReadChunk(f, &chunk) == 0) && (!feof(f))) {
      if (chunk.idnum == LIN_COLOR_24) {
         fread(&material[nummaterials]->color[0], sizeof(UChar_t), 1, f);
         fread(&material[nummaterials]->color[1], sizeof(UChar_t), 1, f);
         fread(&material[nummaterials]->color[2], sizeof(UChar_t), 1, f);
         fseek(f, chunk.endoffset, SEEK_SET);
      }
      else if (chunk.idnum == COLOR_24) {
         fread(&material[nummaterials]->color[0], sizeof(UChar_t), 1, f);
         fread(&material[nummaterials]->color[1], sizeof(UChar_t), 1, f);
         fread(&material[nummaterials]->color[2], sizeof(UChar_t), 1, f);
         fseek(f, chunk.endoffset, SEEK_SET);
      }
      else if (chunk.idnum == LIN_COLOR_F) {
         fread(&fr, sizeof(Float_t), 1, f);
         fread(&fg, sizeof(Float_t), 1, f);
         fread(&fb, sizeof(Float_t), 1, f);
         fseek(f, chunk.endoffset, SEEK_SET);
      }
      else if (chunk.idnum == COLOR_F) {
         fread(&fr, sizeof(Float_t), 1, f);
         fread(&fg, sizeof(Float_t), 1, f);
         fread(&fb, sizeof(Float_t), 1, f);
         fseek(f, chunk.endoffset, SEEK_SET);
      }
      else {
         if (chunk.endoffset < len) {
            //printf("Debug : Unknown Chunk [Mesh Chunk] [0x%x]\n", chunk.idnum);
            fseek(f, chunk.endoffset, SEEK_SET);
         }
         else {
            break;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
Int_t ReadTransparency(FILE *f, UInt_t len)
{
   // reads the Transparency property of the Material Chunk

   Chunk    chunk;
   float    ftransp;
   UShort_t stransp;
   while ((ReadChunk(f, &chunk) == 0) && (!feof(f))) {
      if (chunk.idnum == INT_PERCENTAGE) {
         fread(&stransp, sizeof(UShort_t), 1, f);
         material[nummaterials]->transparency = stransp;
         fseek(f, chunk.endoffset, SEEK_SET);
      }
      else if (chunk.idnum == FLOAT_PERCENTAGE) {
         fread(&ftransp, sizeof(float), 1, f);
         fseek(f, chunk.endoffset, SEEK_SET);
      }
      else {
         if (chunk.endoffset < len) {
            //printf("Debug : Unknown Chunk [Mesh Chunk] [0x%x]\n", chunk.idnum);
            fseek(f, chunk.endoffset, SEEK_SET);
         }
         else {
            break;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
Int_t ReadObjectMaterial(FILE *f)
{
   // reads the name of material associated to the current Chunk

   ReadASCIIZ(f, model.matname);
   return 0;
}

//______________________________________________________________________________
Int_t ReadObjectChunk(FILE *f, UInt_t len)
{
   // reads the Object sub-chunk of the 3D Edit Chunk

   Chunk chunk;
   char name[256];
   ReadASCIIZ(f, name);
   while ((ReadChunk(f, &chunk) == 0) && (!feof(f))) {
      if (chunk.idnum == CHUNK3DOBJECTMESH) {
         ReadMeshChunk(f, chunk.endoffset, name);
      }
      else {
         if (chunk.endoffset < len) {
            //printf("Debug : Unknown Chunk [Object Chunk] [0x%x]\n", chunk.idnum);
            fseek(f, chunk.endoffset, SEEK_SET);
         }
         else {
            break;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
Int_t ReadMeshChunk(FILE *f, UInt_t len, char *objname)
{
   // reads the TriMesh sub-chunk of the Object Chunk

   Chunk chunk;
   model.vlist = 0;
   model.flist = 0;
   model.numverts = model.numfaces = 0;
   sprintf(model.name, "%s", objname);
   printf("Reading Mesh : %s\n", objname);
   while ((ReadChunk(f, &chunk) == 0) && (!feof(f))) {
      if (chunk.idnum == CHUNK3DOBJECTMESHVERTICES) {
         ReadVerticesChunk(f);
      }
      else if (chunk.idnum == CHUNK3DOBJECTMESHFACES) {
         ReadFacesChunk(f);
      }
      else if (chunk.idnum == CHUNK3DOBJECTMESHMAPPING) {
         ReadMappingChunk(f);
      }
      else if (chunk.idnum == CHUNK3DOBJECTMESHMATGROUP) {
         ReadObjectMaterial(f);
      }
      else {
         if (chunk.endoffset < len) {
            //printf("Debug : Unknown Chunk [Mesh Chunk] [0x%x]\n", chunk.idnum);
            fseek(f, chunk.endoffset, SEEK_SET);
         }
         else {
            break;
         }
      }
   }
   ConvertModel();
   if (model.vlist != 0) delete [] model.vlist;
   if (model.flist != 0) delete [] model.flist;
   model.vlist = 0;
   model.flist = 0;
   model.numverts = model.numfaces = 0;
   sprintf(model.name,"");
   nummodels++;
   return 0;
}

//______________________________________________________________________________
Int_t ReadVerticesChunk(FILE *f)
{
   // reads Vertex data of the TriMesh Chunk

   Int_t i;
   UShort_t numv = 0;
   Float_t x, y, z;

   fread(&numv, sizeof(UShort_t), 1, f);
   printf("Reading %i Vertices...", numv);
   model.vlist = new Vertex[numv];
   if (model.vlist == 0) {
      for (i = 0; i < numv; i++) {
         fread(&x, sizeof(Float_t), 1, f);
         fread(&y, sizeof(Float_t), 1, f);
         fread(&z, sizeof(Float_t), 1, f);
      }
      printf("\nWarning : Insufficient Memory to Load Vertices!\n");
      return -1;
   }
   for (i = 0; i < numv; i++) {
      fread(&model.vlist[i].x, sizeof(Float_t), 1, f);
      fread(&model.vlist[i].y, sizeof(Float_t), 1, f);
      fread(&model.vlist[i].z, sizeof(Float_t), 1, f);
   }
   model.numverts = (UInt_t) numv;
   printf("Done!\n");
   return 0;
}

//______________________________________________________________________________
Int_t ReadFacesChunk(FILE *f)
{
   // reads Face data of the TriMesh Chunk

   Int_t i;
   UShort_t numf = 0, v1, v2, v3, attr;

   fread(&numf, sizeof(UShort_t), 1, f);
   printf("Reading %i Faces...", numf);
   model.flist = new Face[numf];
   if (model.flist == 0) {
      for (i = 0; i < numf; i++) {
         fread(&v1, sizeof(UShort_t), 1, f);
         fread(&v2, sizeof(UShort_t), 1, f);
         fread(&v3, sizeof(UShort_t), 1, f);
         fread(&attr, sizeof(UShort_t), 1, f);
      }
      printf("\nWarning : Insufficient Memory to Load Faces!\n");
      return -1;
   }
   for (i = 0; i < numf; i++) {
      fread(&v1, sizeof(UShort_t), 1, f);
      fread(&v2, sizeof(UShort_t), 1, f);
      fread(&v3, sizeof(UShort_t), 1, f);
      fread(&attr, sizeof(UShort_t), 1, f);
      model.flist[i].v1 = (UInt_t)(v1);
      model.flist[i].v2 = (UInt_t)(v2);
      model.flist[i].v3 = (UInt_t)(v3);
   }
   model.numfaces = (UInt_t)(numf);
   printf("Done!\n");
   return 0;
}

//______________________________________________________________________________
Int_t ReadMappingChunk(FILE *f)
{
   // reads Texture Mapping data of the TriMesh Chunk

   UShort_t numuv = 0, i;
   Float_t u, v;

   fread(&numuv, sizeof(UShort_t), 1, f);
   printf("Reading %i Texture Coordinates...", numuv);
   if (numuv != model.numverts) {
      for (i = 0; i < numuv; i++) {
         fread(&u, sizeof(Float_t), 1, f);
         fread(&v, sizeof(Float_t), 1, f);
      }
      printf("\nWarning : Number of Vertices and Mapping Data do not match!\n");
      return -1;
   }
   for (i = 0; i < numuv; i++) {
      fread(&model.vlist[i].u, sizeof(Float_t), 1, f);
      fread(&model.vlist[i].v, sizeof(Float_t), 1, f);
   }
   printf("Done!\n");
   return 0;
}

//______________________________________________________________________________
Int_t ReadASCIIZ(FILE *f, char *name)
{
   // reads a null-terminated string from the given file

   char c = -1;
   Int_t index = 0;

   do {
      fread(&c, sizeof(char), 1, f);
      name[index] = c;
      index++;
      if (index == 255) {
         name[index] = 0;
         c = 0;
      }
   } while ((c != 0) && (!feof(f)));
   return 0;
}

//______________________________________________________________________________
Int_t ConvertModel()
{
   // Convert from Model structure to TEveTriangleSet

   Int_t i;

   ts[nummodels] = new TEveTriangleSet(model.numverts, model.numfaces);
   if (ts[nummodels] == 0)
      return -1;
   for (i=0; i<model.numverts; ++i) {
      ts[nummodels]->SetVertex(i, model.vlist[i].x, model.vlist[i].y,
                               model.vlist[i].z);
   }
   for (i=0; i<model.numfaces; ++i) {
      ts[nummodels]->SetTriangle(i, model.flist[i].v1, model.flist[i].v2,
                                 model.flist[i].v3);
   }
   ts[nummodels]->SetName(model.name);
   ts[nummodels]->SetMainTransparency(0);
   ts[nummodels]->SetMainColor(0);
   for (i = 0; i < nummaterials; i++) {
      if (strcmp(model.matname, material[i]->name) == 0) {
         ts[nummodels]->SetMainTransparency(material[i]->transparency);
         ts[nummodels]->SetMainColorRGB(material[i]->color[0],
                                        material[i]->color[1],
                                        material[i]->color[2]);
         break;
      }
   }
   return 0;
}

//______________________________________________________________________________
void view3ds(const char *fname = "nasashuttle.3ds")
{
   // Main.

   TEveManager::Create();

   Int_t i;
   for (i=0;i<2048;i++) ts[i] = 0;
   for (i=0;i<1024;i++) material[i] = 0;
   model.vlist = 0;
   model.flist = 0;
   nummodels = 0;
   if (Read3DSFile(fname) == 0) {
      TEveTriangleSet* parent = new TEveTriangleSet(0, 0);
      parent->SetName(fname);
      gEve->AddElement(parent);
      for (i=0;i<nummodels;i++) {
         if (ts[i]) {
            ts[i]->GenerateTriangleNormals();
            ts[i]->RefMainTrans().RotateLF(1, 2, TMath::Pi());
            parent->AddElement(ts[i]);
         }
      }
      gEve->Redraw3D(kTRUE);
   }
   for (i = 0; i < nummaterials; i++)
      if (material[i] != 0) delete material[i];
}
