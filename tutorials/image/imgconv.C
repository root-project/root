/// \file
/// \ingroup tutorial_image
/// \notebook -nodraw
/// Open rose512.jpg and save it in the following formats:
///  .png, .gif, .xpm and tiff.
///
/// \macro_code
///
/// \author Valeriy Onuchin

void imgconv()
{
   TImage *img = TImage::Open("$ROOTSYS/tutorials/image/rose512.jpg");
   if (!img) {
      printf("Could not create an image... exit\n");
      return;
   }

   img->WriteImage("rose512.png");
   img->WriteImage("rose512.gif");
   img->WriteImage("rose512.xpm");
   img->WriteImage("rose512.tiff");
}
