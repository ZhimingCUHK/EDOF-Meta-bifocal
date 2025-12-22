import numpy as np

def calculate_dof(focal_length_mm, f_number, focus_dist_mm, pixel_size_um):
    """
    计算景深
    focal_length_mm: 镜头焦距 (25)
    f_number: 光圈值 (2.8)
    focus_dist_mm: 对焦距离 (150)
    pixel_size_um: 像元大小 (3.45)
    """
    # 定义两种 CoC 标准
    coc_strict = pixel_size_um / 1000.0        # 1 pixel (严格物理极限)
    coc_visual = pixel_size_um * 3 / 1000.0    # 3 pixels (视觉可接受范围，类似传统定义)
    
    print(f"--- 参数: 焦距={focal_length_mm}mm, 光圈=F/{f_number}, 对焦距离={focus_dist_mm}mm ---")
    
    for name, coc in [("严格 (1px)", coc_strict), ("宽松 (3px)", coc_visual)]:
        # 1. 超焦距 Hyperfocal distance
        H = (focal_length_mm ** 2) / (f_number * coc)
        
        # 2. 近/远 景深界限
        # 避免分母为0
        if (H - focus_dist_mm) <= 0:
            d_far = float('inf')
        else:
            d_far = (H * focus_dist_mm) / (H - focus_dist_mm)
            
        d_near = (H * focus_dist_mm) / (H + focus_dist_mm)
        
        dof = d_far - d_near
        
        print(f"\n[{name} CoC = {coc*1000:.2f} um]")
        print(f"  超焦距 H: {H/1000:.2f} m")
        print(f"  前景深 Near Limit: {d_near:.2f} mm (-{focus_dist_mm - d_near:.2f} mm)")
        print(f"  后景深 Far Limit : {d_far:.2f} mm (+{d_far - focus_dist_mm:.2f} mm)")
        if d_far == float('inf'):
             print(f"  总景深 DoF       : 无穷大")
        else:
             print(f"  总景深 DoF       : {dof:.2f} mm")

# === Parameter ===
FOCAL_LENGTH = 25    # mm
F_NUMBER = 2.8       # 光圈
FOCUS_DISTANCE = 1000 # mm (0.15m) -> 你的微距工况
PIXEL_SIZE = 12    # um (Sony IMX264)

calculate_dof(FOCAL_LENGTH, F_NUMBER, FOCUS_DISTANCE, PIXEL_SIZE)

