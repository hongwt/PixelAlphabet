"""
高级示例：比较多个字符的逐层特征

这个脚本展示如何：
1. 可视化多个不同字符
2. 分析容易混淆的字符对（如 0 vs Q, 8 vs B）
3. 比较它们在各层的特征差异
"""
from pathlib import Path
from src.inference import CharacterPredictor
import shutil

def visualize_character_comparison():
    """比较容易混淆的字符对"""
    
    checkpoint_path = "checkpoints/best_model.pth"
    
    # 定义要比较的字符对
    confusion_pairs = [
        ('0', 'Q'),  # 数字0 vs 字母Q
        ('8', 'B'),  # 数字8 vs 字母B
        ('1', 'I'),  # 数字1 vs 字母I
        ('5', 'S'),  # 数字5 vs 字母S
    ]
    
    print("="*70)
    print("字符混淆分析 - 逐层特征可视化")
    print("="*70)
    
    # 创建预测器
    print("\n加载模型...")
    predictor = CharacterPredictor(checkpoint_path, device='auto')
    
    # 找到每个字符的示例图片
    test_data_dir = Path("data/test")
    
    for char1, char2 in confusion_pairs:
        print(f"\n{'='*70}")
        print(f"分析字符对: '{char1}' vs '{char2}'")
        print(f"{'='*70}")
        
        # 找图片
        image1_path = None
        image2_path = None
        
        for char_name in [char1, char2]:
            char_dir = test_data_dir / char_name
            if char_dir.exists():
                images = list(char_dir.glob("*.png"))
                if images:
                    if char_name == char1:
                        image1_path = images[0]
                    else:
                        image2_path = images[0]
        
        if image1_path is None or image2_path is None:
            print(f"  ⚠️  未找到字符 '{char1}' 或 '{char2}' 的测试图片，跳过")
            continue
        
        # 预测两个字符
        pred1, conf1 = predictor.predict(image1_path, return_confidence=True)
        pred2, conf2 = predictor.predict(image2_path, return_confidence=True)
        
        print(f"\n字符 '{char1}' 图片: {image1_path.name}")
        print(f"  预测: '{pred1}' (置信度: {conf1:.4f})")
        
        print(f"\n字符 '{char2}' 图片: {image2_path.name}")
        print(f"  预测: '{pred2}' (置信度: {conf2:.4f})")
        
        # 可视化两个字符
        log_dir1 = f"./logs/comparison/{char1}_vs_{char2}/{char1}"
        log_dir2 = f"./logs/comparison/{char1}_vs_{char2}/{char2}"
        
        print(f"\n生成 '{char1}' 的逐层可视化...")
        predictor.visualize_layer_outputs(image1_path, log_dir=log_dir1)
        
        print(f"生成 '{char2}' 的逐层可视化...")
        predictor.visualize_layer_outputs(image2_path, log_dir=log_dir2)
        
        print(f"\n✓ 可视化已保存到:")
        print(f"  {char1}: {log_dir1}")
        print(f"  {char2}: {log_dir2}")
    
    print("\n" + "="*70)
    print("✓ 所有字符对分析完成！")
    print("="*70)
    print("\n要查看和比较可视化结果，运行:")
    print("  tensorboard --logdir logs/comparison")
    print("\n然后在 TensorBoard 中:")
    print("  1. 访问 http://localhost:6006")
    print("  2. 在左侧菜单选择不同的运行（runs）")
    print("  3. 比较相同层的特征图，观察差异")
    print("  4. 特别关注容易混淆的字符在各层的不同表现")
    print()

def visualize_all_classes():
    """为每个类别生成一个样本的可视化"""
    
    checkpoint_path = "checkpoints/best_model.pth"
    test_data_dir = Path("data/test")
    
    print("="*70)
    print("所有类别样本可视化")
    print("="*70)
    
    predictor = CharacterPredictor(checkpoint_path, device='auto')
    
    # 获取所有类别
    classes = sorted([d.name for d in test_data_dir.iterdir() 
                     if d.is_dir() and d.name != 'NA'])
    
    print(f"\n找到 {len(classes)} 个类别")
    
    visualized = 0
    for class_name in classes:
        class_dir = test_data_dir / class_name
        images = list(class_dir.glob("*.png"))
        
        if not images:
            continue
        
        # 使用第一张图片
        image_path = images[0]
        
        # 预测
        pred, conf = predictor.predict(image_path, return_confidence=True)
        
        # 可视化
        log_dir = f"./logs/all_classes/{class_name}"
        predictor.visualize_layer_outputs(image_path, log_dir=log_dir)
        
        visualized += 1
        print(f"  [{visualized}/{len(classes)}] '{class_name}': 预测='{pred}' (置信度={conf:.4f})")
    
    print("\n" + "="*70)
    print(f"✓ 完成！已为 {visualized} 个类别生成可视化")
    print("="*70)
    print("\n要查看结果:")
    print("  tensorboard --logdir logs/all_classes")
    print()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='字符特征对比分析')
    parser.add_argument('--mode', type=str, default='confusion',
                        choices=['confusion', 'all'],
                        help='分析模式: confusion=混淆字符对比, all=所有类别')
    
    args = parser.parse_args()
    
    if args.mode == 'confusion':
        visualize_character_comparison()
    elif args.mode == 'all':
        visualize_all_classes()

if __name__ == '__main__':
    main()
