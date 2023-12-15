import itk
from .lung_ct import init_network

def brain_network_preprocess(image: "itk.Image") -> "itk.Image":
    if type(image) == itk.Image[itk.SS, 3] :
        cast_filter = itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.F, 3]].New()
        cast_filter.SetInput(image)
        cast_filter.Update()
        image = cast_filter.GetOutput()
    _, max_ = itk.image_intensity_min_max(image)
    image = itk.shift_scale_image_filter(image, shift=0., scale = .9 / max_)
    return image

def brain_registration_model(pretrained=True):
    return init_network("brain", pretrained=pretrained)
