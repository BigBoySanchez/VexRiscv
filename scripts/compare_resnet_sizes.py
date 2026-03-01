import torch
import torchvision.models as models
import math

def main():
    print(f"{'Model Name':<30} | {'Float32 Size (MB)':<18} | {'INT8 Size (MB)':<15} | {'BlockDialect Size (MB)':<22} | {'Compression Ratio':<18}")
    print("-" * 115)
    
    all_models = models.list_models()
    resnets = [m for m in all_models if 'resne' in m.lower() and not m.startswith('quantized_')]
    
    for name in resnets:
        try:
            model = models.get_model(name, weights=None)
        except Exception as e:
            continue
            
        float_bytes = 0
        int8_bytes = 0
        bd_bytes = 0
        
        for p_name, param in model.named_parameters():
            if "weight" in p_name or "bias" in p_name:
                numel = param.numel()
                float_bytes += numel * 4
                int8_bytes += numel
                
                blocks = math.ceil(numel / 32.0)
                bd_bytes += blocks * 18
                
        float_mb = float_bytes / (1024 * 1024)
        int8_mb = int8_bytes / (1024 * 1024)
        bd_mb = bd_bytes / (1024 * 1024)
        compression_ratio = float_mb / bd_mb if bd_mb > 0 else 0
        
        print(f"{name:<30} | {float_mb:>15.2f} MB | {int8_mb:>12.2f} MB | {bd_mb:>19.2f} MB | {compression_ratio:>17.2f}x")

if __name__ == '__main__':
    main()
