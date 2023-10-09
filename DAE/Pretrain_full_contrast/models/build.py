from .simmim_advanced import build_simmim
from .swin_transformer import build_swin
from .vision_transformer import build_vit


def build_model(args, is_pretrain=True):
    if is_pretrain:
        model = build_simmim(args)
    else:
        model_type = args.model_type
        if model_type == "swin":
            model = build_swin(args)
        elif model_type == "vit":
            model = build_vit(args)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model
