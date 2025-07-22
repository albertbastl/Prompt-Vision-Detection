
from transformers import AutoConfig

SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"

def get_model_config():
    config = AutoConfig.from_pretrained(SIGLIP_MODEL_ID)
    
    vision_config = config.vision_config
    
    patch_grid_size = vision_config.image_size // vision_config.patch_size
    
    print(f"-> Model: {SIGLIP_MODEL_ID}")
    print(f"-> Image Size: {vision_config.image_size}")
    print(f"-> Patch Size: {vision_config.patch_size}")
    print(f"-> Derived Patch Grid Size: {patch_grid_size}x{patch_grid_size}")
    
    return SIGLIP_MODEL_ID, patch_grid_size, vision_config.image_size

MODEL_ID, PATCH_GRID_SIZE, IMAGE_SIZE = get_model_config()
TEXT_MAX_LENGTH = 32
