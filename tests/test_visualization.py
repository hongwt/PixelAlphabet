"""
快速测试：验证逐层可视化功能
"""
import sys
from pathlib import Path

# 检查必要的依赖
try:
    import torch
    import matplotlib
    from torch.utils.tensorboard import SummaryWriter
    print("✓ 所有必要的库都已安装")
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)

# 检查模型检查点
checkpoint_path = Path("checkpoints/best_model.pth")
if not checkpoint_path.exists():
    print(f"❌ 模型检查点不存在: {checkpoint_path}")
    sys.exit(1)

print(f"✓ 找到模型检查点: {checkpoint_path}")

# 检查测试图片
test_data_dir = Path("data/test")
test_image = None
for class_dir in test_data_dir.iterdir():
    if class_dir.is_dir() and class_dir.name != 'NA':
        images = list(class_dir.glob("*.png"))
        if images:
            test_image = images[0]
            break

if test_image is None:
    print("❌ 未找到测试图片")
    sys.exit(1)

print(f"✓ 找到测试图片: {test_image}")

# 导入并测试
try:
    from src.inference import CharacterPredictor
    print("✓ 成功导入 CharacterPredictor")
    
    # 创建预测器
    print("\n加载模型...")
    predictor = CharacterPredictor(str(checkpoint_path), device='cpu')
    
    # 测试预测
    print(f"\n测试预测功能...")
    char, conf = predictor.predict(test_image, return_confidence=True)
    print(f"✓ 预测结果: '{char}' (置信度: {conf:.4f})")
    
    # 测试可视化
    print(f"\n测试逐层可视化功能...")
    log_dir = predictor.visualize_layer_outputs(
        test_image,
        log_dir="./logs/test_visualization"
    )
    print(f"✓ 可视化成功!")
    
    print("\n" + "="*60)
    print("✓ 所有测试通过！")
    print("="*60)
    print(f"\n要查看可视化结果，请运行:")
    print(f"  tensorboard --logdir {log_dir}")
    print(f"\n然后在浏览器中打开: http://localhost:6006")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
