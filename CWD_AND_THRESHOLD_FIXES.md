# CWD-based IMNF Robustness and Energy Threshold Extension - Implementation Summary

## 问题总结 (Problem Summary)

本次更新解决了两个核心问题：

### 1. IMNF中CWD计算的稳定性问题 ✅

**问题描述**：
- IMNF利用CWD（Choi-Williams Distribution）的计算会因为各种异常不断回退到普通的MNF
- CW变换算法并没有正常运作

**解决方案**：
- 在`_compute_choi_williams_distribution()`中添加了全面的输入验证
- 实现了特定的错误处理（ValueError用于无效输入，RuntimeError用于计算失败）
- 改进了IMNF计算逻辑，只在CWD真正无法计算时才回退到MNF
- 修复了权重索引bug，确保权重与计算的IMF值正确对齐
- 添加了模块级常量以提高可维护性

**结果**：
- CWD现在可以可靠地处理正常信号，同时优雅地处理边缘情况
- IMNF在可能的情况下使用真正的CWD计算，只在必要时（信号过短、能量过低）回退到MNF

### 2. 能量阈值调节范围扩展 ✅

**问题描述**：
- sEMG收缩事件的瞬时能量水平差异可以非常巨大
- 在time-integrated energy profile图片中，需要扩展能量阈值的调节范围
- 需要允许用户在极端情况下将阈值降至非常贴近极限静息态的水平

**解决方案**：
- 将GUI滑块范围从[0.5, 0.9]扩展到**[0.1, 0.95]**
- 更新了HHT检测常量：
  - HHT_MIN_ENERGY_THRESHOLD: 0.3 → **0.1** (第10百分位 - 非常接近基线)
  - HHT_NOISE_FLOOR_PERCENTILE: 5 → **3**
  - HHT_MAX_THRESHOLD_PERCENTILE: 60 → **70**
  - HHT_MIN_COMPACTNESS: 0.1 → **0.05**

**结果**：
- 用户现在可以通过将阈值降至非常接近基线静息态水平来检测极端情况下的微弱肌肉活动
- 同时保持了严格检测的能力（高达第95百分位）

## 技术实现细节 (Technical Implementation Details)

### CWD计算增强

```python
# 添加的输入验证
- 验证信号为非空1D数组
- 要求最小信号长度（16个采样点）
- 检查信号具有足够能量（> EPSILON）
- 验证采样频率和sigma为正数
- 对n_freq_bins和n_time_bins设置合理边界

# 添加的模块级常量
MIN_CWD_SIGNAL_LENGTH = 16
MIN_CWD_TIME_BINS = 32
DEFAULT_CWD_TIME_BINS = 128
MIN_CWD_FREQ_BINS = 64
DEFAULT_CWD_FREQ_BINS = 256
```

### IMNF计算改进

```python
# 改进的错误处理逻辑
1. 将IMNF初始化为MNF作为安全默认值
2. 在尝试CWD前检查信号长度（< 16个采样点则跳过）
3. 分别捕获特定的CWD异常（ValueError, RuntimeError）
4. 验证中间结果（频率掩码、IMF值）
5. 跟踪有效时间索引以确保权重正确对齐
6. 只有在所有验证通过时才使用CWD结果
7. 添加兜底异常处理（保持MNF默认值）
```

### 能量阈值范围扩展

**GUI变更**：
- 单次检测和批量检测标签页的能量阈值滑块都扩展到[0.1, 0.95]
- 更新了提示文本："Extended range: 0.1=very sensitive, 0.95=very strict"

**常量变更**：
```python
# detection.py中的HHT常量更新
HHT_MIN_ENERGY_THRESHOLD: 0.3 → 0.1  # 允许非常灵敏的检测
HHT_NOISE_FLOOR_PERCENTILE: 5 → 3   # 降低底噪层以实现极端灵敏度
HHT_MAX_THRESHOLD_PERCENTILE: 60 → 70  # 更宽的上限
HHT_MIN_COMPACTNESS: 0.1 → 0.05     # 更广泛的检测
```

## 测试验证 (Testing and Validation)

### 测试覆盖率

**CWD边缘情况**：
- ✅ 短信号（< 16采样点）- 正确以ValueError拒绝
- ✅ 接近零能量信号 - 正确以ValueError拒绝
- ✅ 正常信号 - 成功处理并产生有限值
- ✅ 无效参数（负fs、零sigma）- 正确拒绝
- ✅ IMNF回退行为 - 在需要时优雅地使用MNF

**极端能量阈值**：
- ✅ 非常低的阈值(0.1) - 检测微弱活动
- ✅ 非常高的阈值(0.9) - 选择性检测
- ✅ 阈值关系 - 较低阈值检测的持续时间≥较高阈值

### 所有测试通过 ✅

```
✅ test_cwd_and_export_optimization.py - 原始CWD测试
✅ test_cwd_robustness.py - 新的边缘情况测试（8个测试用例）
✅ tests/test_basic.py - 核心功能测试
✅ tests/test_enhanced_features.py - 增强功能测试（5个测试用例）
```

### 代码质量 ✅

```
✅ 代码审查完成 - 所有重要问题已解决
✅ 安全扫描（CodeQL）- 0个警报
✅ 向后兼容性 - 无破坏性更改
```

## 使用指南 (Usage Guide)

### 对于CWD/IMNF用户

现在IMNF计算更可靠了：
- 正常长度信号（≥16采样点）：使用完整的CWD计算
- 短信号（<16采样点）：自动回退到MNF
- 低能量信号：自动回退到MNF
- 所有情况都能得到有效的频率值

### 对于极端情况检测

现在可以检测非常微弱的肌肉活动：

1. **打开GUI应用**
2. **导航到检测设置**
3. **调整能量阈值滑块**：
   - **0.1-0.3**: 非常灵敏，接近静息态基线，适合检测微弱活动
   - **0.4-0.6**: 平衡检测（默认0.65）
   - **0.7-0.95**: 严格检测，只检测强烈活动

4. **观察time-integrated energy profile**：
   - detected数值代表检测到的能量水平
   - 现在可以将阈值降至非常接近基线的水平

### 代码示例

```python
from semg_preprocessing.hht import extract_semg_features
from semg_preprocessing.detection import detect_activity_hht

# IMNF特征提取 - 现在更可靠
signal = ...  # 你的sEMG信号
fs = 1000.0   # 采样频率
features = extract_semg_features(signal, fs)
imnf = features['IMNF']  # 使用CWD或在必要时回退到MNF

# HHT检测，使用扩展的能量阈值范围
segments = detect_activity_hht(
    signal, 
    fs=fs,
    energy_threshold=0.1,  # 非常灵敏 - 新功能！
    min_duration=0.1
)
```

## 影响评估 (Impact Assessment)

### 积极影响

1. **可靠性提升**：CWD计算现在有适当的错误处理和验证
2. **灵活性增强**：能量阈值范围扩展允许检测极端情况
3. **用户体验改善**：清晰的错误消息和优雅的回退
4. **维护性提高**：添加了命名常量和更好的代码组织

### 无破坏性影响

- ✅ 默认值保持不变（能量阈值0.65）
- ✅ API未更改
- ✅ 现有工作流继续工作
- ✅ 所有现有测试通过

## 未来工作建议 (Future Work Recommendations)

虽然当前实现已经解决了问题，但以下增强可能在未来有价值：

1. **日志记录**：添加可选的日志记录以跟踪CWD回退事件
2. **性能优化**：对于非常长的信号优化CWD计算
3. **可视化**：在GUI中添加CWD使用指示器
4. **文档**：添加更多关于何时使用极端阈值的用户指南

## 结论 (Conclusion)

两个问题都已成功解决：

✅ **问题1**：CWD计算现在稳健且正常运行，具有适当的错误处理
✅ **问题2**：能量阈值范围扩展，允许检测非常接近基线静息态的活动

实现遵循指定的指导原则：
- CW变换仍然是IMNF计算的主要方法
- 稳健的错误处理防止了持续回退到MNF
- 扩展的阈值范围支持极端情况检测
- 保留了所有现有功能
