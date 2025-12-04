#!/usr/bin/env python3
"""
详细分析各个领域的Reward计算过程和结果
"""
import sys
sys.path.append('/root/work/Agentic-RL-Scaling-Law')

import pandas as pd
import json
from src.reward.guru_reward_improved import compute_score, extract_answer, normalize_answer
from pathlib import Path

def analyze_domain_rewards():
    
    domains = [
        ('math', 'math__combined_54.4k.parquet'),
        ('logic', 'logic__arcagi1_111.parquet'), 
        ('code', 'codegen__leetcode2k_1.3k.parquet'),
        ('stem', 'stem__web_3.6k.parquet')
    ]
    
    base_dir = Path('../../data/guru_verl/train')
    
    print("="*120)

    for domain_name, filename in domains:
        filepath = base_dir / filename
        if not filepath.exists():
            print(f"\n❌ {domain_name.upper()} - 文件不存在: {filename}")
            continue
            
        print(f"\n{'🧮' if domain_name=='math' else '💻' if domain_name=='code' else '🧩' if domain_name=='logic' else '🔬'} {domain_name.upper()} 领域分析")
        print("="*80)
        
        df = pd.read_parquet(filepath)
        print(f"📊 数据集大小: {len(df)} 样本")
        
        # 选择2-3个有代表性的样本进行详细分析
        sample_indices = [0, len(df)//4, len(df)//2] if len(df) > 3 else list(range(len(df)))
        
        for i, idx in enumerate(sample_indices[:3]):  # 最多分析3个样本
            print(f"\n--- 样本 {i+1} (索引: {idx}) ---")
            sample = df.iloc[idx]
            
            # 1. 显示问题
            if 'prompt' in sample and len(sample['prompt']) > 0:
                problem = sample['prompt'][0]['content']
                print(f"📝 问题: {problem[:200]}...")
            
            # 2. 显示ground truth
            ground_truth = sample['reward_model']['ground_truth']
            print(f"🎯 Ground Truth类型: {type(ground_truth)}")
            
            if domain_name == 'code' and isinstance(ground_truth, str):
                try:
                    gt_dict = json.loads(ground_truth)
                    if 'functional' in gt_dict:
                        print(f"🧪 测试函数:\n{gt_dict['functional'][:300]}...")
                except:
                    print(f"🎯 Ground Truth: {str(ground_truth)[:200]}...")
            else:
                if len(str(ground_truth)) > 200:
                    print(f"🎯 Ground Truth: {str(ground_truth)[:200]}...")
                else:
                    print(f"🎯 Ground Truth: {ground_truth}")
            
            # 3. 显示数据集中的答案
            # 现在所有领域的ground truth都在reward_model中
            if 'ground_truth' in sample['reward_model']:
                dataset_answer = sample['reward_model']['ground_truth']
                
                if domain_name == 'code':
                    # Code领域的ground_truth是测试代码，创建一个简单的Solution作为答案
                    # 这里只是为了演示，实际训练时会生成真实的代码
                    formatted_solution = """```python
class Solution:
    def solve(self):
        # This is a placeholder solution for demonstration
        return None
```"""
                    print(f"💡 演示答案: [使用占位符代码]")
                else:
                    formatted_solution = dataset_answer
                    if len(str(dataset_answer)) > 200:
                        print(f"💡 数据集真实答案: {str(dataset_answer)[:200]}...")
                    else:
                        print(f"💡 数据集真实答案: {dataset_answer}")
            else:
                # 创建一个模拟的错误答案进行测试
                dataset_answer = "This is a test wrong answer"
                formatted_solution = dataset_answer
                print(f"💡 模拟测试答案: {dataset_answer}")
            
            # 4. 详细计算过程
            print(f"\n🔍 Reward计算过程:")
            
            # 4.1 答案提取
            extracted = extract_answer(formatted_solution, domain_name)
            print(f"   步骤1 - 答案提取:")
            if len(str(extracted)) > 150:
                print(f"     提取结果: {str(extracted)[:150]}...")
            else:
                print(f"     提取结果: {extracted}")
            
            # 4.2 答案标准化
            if domain_name != 'code':  # code domain不需要标准化比较
                normalized_solution = normalize_answer(extracted, domain_name)
                normalized_truth = normalize_answer(ground_truth, domain_name)
                print(f"   步骤2 - 答案标准化:")
                print(f"     标准化解答: {normalized_solution}")
                print(f"     标准化真值: {normalized_truth}")
            
            # 4.3 最终评分
            try:
                score = compute_score(
                    formatted_solution,
                    ground_truth,
                    data_source=sample.get('data_source', ''),
                    domain=domain_name
                )
                print(f"   步骤3 - 最终评分: {score:.4f}")
                
                # 解释评分结果
                if domain_name == 'math':
                    if score >= 0.99:
                        print(f"     ✅ 数学答案正确 (使用VeRL math_score或精确匹配)")
                    else:
                        print(f"     ❌ 数学答案错误")
                        
                elif domain_name == 'code':
                    if score >= 0.99:
                        print(f"     ✅ 代码通过所有测试用例")
                    elif score > 0:
                        total_tests = len([line for line in str(ground_truth) if 'assert' in line])
                        passed_tests = int(score * total_tests)
                        print(f"     🔶 代码部分正确: {passed_tests}/{total_tests} 测试用例通过")
                    else:
                        print(f"     ❌ 代码错误 (语法错误、结构错误或所有测试失败)")
                        
                elif domain_name == 'logic':
                    if score >= 0.99:
                        print(f"     ✅ 逻辑答案完全匹配")
                    else:
                        print(f"     ❌ 逻辑答案不匹配")
                        
                elif domain_name == 'stem':
                    if score >= 0.99:
                        print(f"     ✅ 科学答案正确 (数学评分或精确匹配)")
                    else:
                        print(f"     ❌ 科学答案错误")
                
            except Exception as e:
                print(f"   ❌ 评分计算出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 统计该领域的整体表现
        print(f"\n📈 {domain_name.upper()} 领域整体统计 (前50个样本):")
        test_count = min(50, len(df))
        scores = []
        
        for idx in range(test_count):
            try:
                sample = df.iloc[idx]
                ground_truth = sample['reward_model']['ground_truth']
                
                if domain_name == 'code':
                    # Code领域使用占位符代码测试（实际会是0分）
                    solution = """```python
class Solution:
    def solve(self):
        return None
```"""
                else:
                    # 对于其他领域，我们用ground_truth作为"正确答案"来测试
                    # 这样应该得到高分，验证reward函数工作正常
                    solution = ground_truth
                
                score = compute_score(
                    solution,
                    ground_truth,
                    data_source=sample.get('data_source', ''),
                    domain=domain_name
                )
                scores.append(score)
            except Exception as e:
                # print(f"Error processing sample {idx}: {e}")
                pass
        
        if scores:
            avg_score = sum(scores) / len(scores)
            perfect_count = sum(1 for s in scores if s >= 0.99)
            partial_count = sum(1 for s in scores if 0.1 <= s < 0.99)
            zero_count = sum(1 for s in scores if s < 0.1)
            
            print(f"   有效样本数: {len(scores)}/{test_count}")
            print(f"   平均得分: {avg_score:.4f}")
            print(f"   完美得分(≥0.99): {perfect_count} ({perfect_count/len(scores)*100:.1f}%)")
            print(f"   部分得分(0.1-0.99): {partial_count} ({partial_count/len(scores)*100:.1f}%)")
            print(f"   零分或接近零分(<0.1): {zero_count} ({zero_count/len(scores)*100:.1f}%)")

def explain_reward_mechanisms():
    """解释各领域的reward计算机制"""
    
    print(f"\n{'='*120}")
    print("各领域Reward计算机制原理说明")
    print("="*120)
    
    mechanisms = {
        "🧮 MATH领域": {
            "计算方式": [
                "1. 优先使用VeRL内置的math_score()函数进行数学表达式评估",
                "2. 自动识别\\boxed{答案}格式和'final answer'、'answer is'等模式",
                "3. 如果math_score()失败，回退到标准化字符串比较",
                "4. 支持数学符号、分数、方程式等复杂格式"
            ],
            "评分规则": "二元评分: 1.0(正确) 或 0.0(错误)",
            "优势": "专门针对数学问题优化，能处理各种数学表达式格式"
        },
        
        "💻 CODE领域": {
            "计算方式": [
                "1. 从markdown代码块中提取Python代码",
                "2. 验证代码结构(必须包含class Solution和方法定义)",
                "3. 在安全的受限环境中执行代码",
                "4. 解析JSON格式的单元测试并逐个执行",
                "5. 计算通过的测试用例比例作为分数"
            ],
            "评分规则": "梯度评分: 0.0-1.0 (通过的测试用例比例)",
            "优势": "实际执行代码和测试，提供精确的功能正确性评估"
        },
        
        "🧩 LOGIC领域": {
            "计算方式": [
                "1. 使用正则表达式匹配结构化答案模式",
                "2. 识别'answer:'、'therefore'、结论等关键词",
                "3. 标准化布尔值答案(Yes/No, True/False等)",
                "4. 进行精确字符串匹配比较"
            ],
            "评分规则": "二元评分: 1.0(完全匹配) 或 0.0(不匹配)",
            "优势": "针对逻辑推理问题的结构化答案格式优化"
        },
        
        "🔬 STEM/SCIENCE领域": {
            "计算方式": [
                "1. 首先尝试使用math_score()处理数值型科学问题",
                "2. 支持科学计数法、物理公式等格式",
                "3. 如果数学评分失败，回退到精确字符串匹配",
                "4. 处理方程式、物理量、化学式等科学表达式"
            ],
            "评分规则": "二元评分: 1.0(正确) 或 0.0(错误)",
            "优势": "结合数学评分和精确匹配，适应科学问题的多样性"
        }
    }
    
    for domain, info in mechanisms.items():
        print(f"\n{domain}")
        print("-" * 60)
        print("🔧 计算方式:")
        for step in info["计算方式"]:
            print(f"   {step}")
        print(f"📊 评分规则: {info['评分规则']}")
        print(f"✨ 优势: {info['优势']}")

if __name__ == "__main__":
    analyze_domain_rewards()
    explain_reward_mechanisms()
    
    print(f"\n{'='*120}")
    print("关键改进总结")
    print("="*120)
    print("""
🚀 主要改进:
1. CODE领域: 从完全无效(总是0分) → 实际执行单元测试的梯度评分系统
2. 安全执行: 在受限环境中安全执行用户代码，防止恶意操作
3. 部分正确支持: CODE领域支持0.0-1.0的连续评分
4. 鲁棒性增强: 更好的错误处理和回退机制
5. 答案提取优化: 改进各领域的答案模式识别

📈 影响:
- 提升PPO训练中CODE领域的reward信号质量
- 为部分正确的解答提供合理的梯度反馈
- 增强整体训练稳定性和收敛效果
""")