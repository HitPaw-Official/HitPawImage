from .models import ViTAE_noRC_MaxPooling_bias_basic_stages4_14
from .models import ViTAE_noRC_MaxPooling_DecoderV1
from .models_tiny import ViTAE_noRC_MaxPooling_bias_basic_stages4_14_tiny
from .models_tiny import ViTAE_noRC_MaxPooling_DecoderV1_tiny
from .models4channel import ViTAE_noRC_MaxPooling_bias_basic_stages4_14_4channel, ViTAE_noRC_MaxPooling_DecoderV1_4channel
from .models import ViTAE_noRC_MaxPooling_Matting


__all__ = ['p3mnet_vitae_s', 'p3mnet_vitae_s_tiny', 'p3mnet_vitae_s_4channel']


def p3mnet_vitae_s(pretrained=True, **kwargs):
    encoder = ViTAE_noRC_MaxPooling_bias_basic_stages4_14(pretrained=pretrained, **kwargs)
    decoder = ViTAE_noRC_MaxPooling_DecoderV1()
    model = ViTAE_noRC_MaxPooling_Matting(encoder, decoder)
    return model


def p3mnet_vitae_s_tiny(pretrained=False, **kwargs):
    encoder = ViTAE_noRC_MaxPooling_bias_basic_stages4_14_tiny(pretrained=pretrained, **kwargs)
    decoder = ViTAE_noRC_MaxPooling_DecoderV1_tiny()
    model = ViTAE_noRC_MaxPooling_Matting(encoder, decoder)
    return model

def p3mnet_vitae_s_4channel(pretrained=False, **kwargs):
    encoder = ViTAE_noRC_MaxPooling_bias_basic_stages4_14_4channel(pretrained=pretrained, **kwargs)
    decoder = ViTAE_noRC_MaxPooling_DecoderV1_4channel()
    model = ViTAE_noRC_MaxPooling_Matting(encoder, decoder)
    return model
