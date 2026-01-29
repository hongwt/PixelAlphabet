"""
示例脚本：可视化模型每一层的输出

使用方法：
    python visualize_layers.py

这个脚本会：
1. 加载训练好的模型
2. 选择一张测试图片
3. 可视化图片经过每一层后的特征图
4. 在TensorBoard中显示结果
"""
from pathlib import Path
from src.inference import CharacterPredictor

def main():
    # 配置
    checkpoint_path = "checkpoints/best_model.pth"
    
    # 找一张测试图片
    test_data_dir = Path("data/test")
    
    # 尝试找到一张测试图片
    test_image = None
    for class_dir in test_data_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.png"))
            if images:
                test_image = images[0]
                break
    
    if test_image is None:
        print("❌ 未找到测试图片")
        return
    
    print(f"使用测试图片: {test_image}")
    
    # 创建预测器
    print("\n加载模型...")
    predictor = CharacterPredictor(checkpoint_path, device='auto')
    
    # 执行预测
    print("\n执行预测...")
    char, confidence = predictor.predict(test_image, return_confidence=True)
    print(f"预测结果: '{char}' (置信度: {confidence:.4f})")
    
    # 可视化各层输出
    print("\n生成逐层可视化...")
    log_dir = predictor.visualize_layer_outputs(
        test_image, 
        log_dir="./logs/layer_visualization"
    )
    
    print("\n" + "="*60)
    print("✓ 可视化完成！")
    print("="*60)
    print(f"\n要查看可视化结果，请运行:")
    print(f"  tensorboard --logdir {log_dir}")
    print(f"\n然后在浏览器中打开: http://localhost:6006")
    print("\n在TensorBoard中你可以看到:")
    print("  • IMAGES 标签: 每一层的特征图（通道可视化）")
    print("  • GRAPHS 标签: 模型的网络结构图")
    print("  • 每一层输出的详细信息和形状")
    print()

if __name__ == '__main__':
    main()
