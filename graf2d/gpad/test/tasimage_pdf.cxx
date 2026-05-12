#include "gtest/gtest.h"

#include "TCanvas.h"
#include "TImage.h"
#include "TString.h"
#include "TSystem.h"

#include <fstream>
#include <string>
#include <vector>
#include <zlib.h>

namespace {

// Read a binary file fully into a string.
std::string SlurpFile(const TString &path)
{
   std::ifstream in(path.Data(), std::ios::binary);
   std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
   return data;
}

// FlateDecode the byte range [begin, end) of `pdf`. Returns an empty string if
// the range is not a valid zlib stream.
std::string Inflate(const std::string &pdf, std::size_t begin, std::size_t end)
{
   std::string out;
   if (end <= begin)
      return out;
   z_stream zs{};
   zs.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(pdf.data() + begin));
   zs.avail_in = static_cast<uInt>(end - begin);
   if (inflateInit(&zs) != Z_OK)
      return out;
   std::vector<char> buf(64 * 1024);
   int rc = Z_OK;
   do {
      zs.next_out = reinterpret_cast<Bytef *>(buf.data());
      zs.avail_out = static_cast<uInt>(buf.size());
      rc = inflate(&zs, Z_NO_FLUSH);
      if (rc == Z_OK || rc == Z_STREAM_END)
         out.append(buf.data(), buf.size() - zs.avail_out);
      else
         break;
   } while (rc != Z_STREAM_END);
   inflateEnd(&zs);
   return out;
}

// Locate the data range of the "stream ... endstream" block that starts at or
// after `from`. Returns false if none is found.
bool FindStream(const std::string &pdf, std::size_t from, std::size_t &dataBegin, std::size_t &dataEnd,
                std::size_t &next)
{
   std::size_t s = pdf.find("stream", from);
   if (s == std::string::npos)
      return false;
   dataBegin = s + 6; // strlen("stream")
   if (dataBegin < pdf.size() && pdf[dataBegin] == '\r')
      ++dataBegin;
   if (dataBegin < pdf.size() && pdf[dataBegin] == '\n')
      ++dataBegin;
   std::size_t e = pdf.find("endstream", dataBegin);
   if (e == std::string::npos)
      return false;
   dataEnd = e;
   if (dataEnd > dataBegin && pdf[dataEnd - 1] == '\n')
      --dataEnd;
   if (dataEnd > dataBegin && pdf[dataEnd - 1] == '\r')
      --dataEnd;
   next = e + 9; // strlen("endstream")
   return true;
}

// FlateDecode every stream in the PDF and concatenate the results. The page
// content stream (which carries the painting operators) is among them.
std::string DecodeAllFlateStreams(const std::string &pdf)
{
   std::string out;
   std::size_t pos = 0, b, e, next;
   while (FindStream(pdf, pos, b, e, next)) {
      out += Inflate(pdf, b, e);
      pos = next;
   }
   return out;
}

// FlateDecode the stream of the first image XObject in the PDF.
std::string DecodeImageXObject(const std::string &pdf)
{
   std::size_t img = pdf.find("/Subtype /Image");
   if (img == std::string::npos)
      return {};
   std::size_t b, e, next;
   if (!FindStream(pdf, img, b, e, next))
      return {};
   return Inflate(pdf, b, e);
}

} // namespace

// Render a small synthetic TImage to a PDF and verify it is embedded as a
// proper PDF image XObject: a "/Subtype /Image" object carrying the pixels,
// referenced from the page content stream by a "/ImN Do" operator. The
// embedded pixels are decoded back and checked against the four colours that
// were drawn, so the test guards pixel fidelity, not just the PDF structure.
//
// This covers the TASImage::Paint -> TPDF::CellArray* path: before the fix it
// emitted only a "PDF not implemented yet" warning and no image data.
TEST(TASImage, PDFEmbedsImageXObject)
{
   const TString pdfFile = "tasimage_pdf_embed.pdf";

   TImage *img = TImage::Create();
   ASSERT_NE(img, nullptr) << "TImage::Create failed (ASImage plugin missing?)";
   // The first FillRectangle sizes the (initially empty) image, so it must
   // cover the whole 80x80 area; the other three then paint the quadrants.
   img->FillRectangle("#ff0000", 0, 0, 80, 80);   // red base / top-left
   img->FillRectangle("#00ff00", 40, 0, 40, 40);  // top-right green
   img->FillRectangle("#0000ff", 0, 40, 40, 40);  // bottom-left blue
   img->FillRectangle("#ffff00", 40, 40, 40, 40); // bottom-right yellow

   TCanvas c("tasimage_pdf_canvas", "tasimage_pdf_canvas", 300, 300);
   img->Draw("X");
   c.SaveAs(pdfFile);
   delete img;

   FileStat_t st;
   ASSERT_EQ(gSystem->GetPathInfo(pdfFile, st), 0) << "PDF file was not created.";
   ASSERT_GT(st.fSize, 1024) << "PDF is suspiciously small.";

   const std::string pdf = SlurpFile(pdfFile);
   ASSERT_FALSE(pdf.empty());
   ASSERT_EQ(pdf.compare(0, 4, "%PDF"), 0) << "File does not look like a PDF.";

   // The bitmap must be stored as its own image XObject, not dropped.
   EXPECT_NE(pdf.find("/Subtype /Image"), std::string::npos)
      << "No image XObject in the PDF — TASImage::Paint did not embed the bitmap.";
   EXPECT_NE(pdf.find("/ColorSpace /DeviceRGB"), std::string::npos)
      << "Image XObject colour space declaration missing.";
   EXPECT_NE(pdf.find("/BitsPerComponent 8"), std::string::npos) << "Image XObject bit depth declaration missing.";

   // The page content stream must actually paint that XObject.
   const std::string content = DecodeAllFlateStreams(pdf);
   ASSERT_FALSE(content.empty()) << "No Flate streams could be decoded.";
   EXPECT_NE(content.find("/Im1 Do"), std::string::npos)
      << "Page content stream does not paint the image XObject (no '/Im1 Do').";

   // The embedded pixels must match what was drawn: all four colours present.
   const std::string pixels = DecodeImageXObject(pdf);
   ASSERT_FALSE(pixels.empty()) << "Image XObject stream did not FlateDecode.";
   EXPECT_EQ(pixels.size() % 3u, 0u) << "DeviceRGB data is not a whole number of pixels.";
   EXPECT_NE(pixels.find(std::string("\xff\x00\x00", 3)), std::string::npos) << "red missing";
   EXPECT_NE(pixels.find(std::string("\x00\xff\x00", 3)), std::string::npos) << "green missing";
   EXPECT_NE(pixels.find(std::string("\x00\x00\xff", 3)), std::string::npos) << "blue missing";
   EXPECT_NE(pixels.find(std::string("\xff\xff\x00", 3)), std::string::npos) << "yellow missing";

   gSystem->Unlink(pdfFile);
}
