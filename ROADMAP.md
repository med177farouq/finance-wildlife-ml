# FinNeX Project Roadmap - MSc Research Plan
**Multi-Modal Image-Based CNN for Tesla Stock Prediction**  
*Based on Project Analysis Chat - August 2025*

## 🎯 Project Overview
**FinNeX (Financial Neural Exchange)**: A framework that dynamically fuses price trends, macroeconomic indicators, news sentiment, and company fundamentals into unified Mega-Images for CNN-based stock prediction.

**Research Question**: Can integrating multi-modal financial data into structured image-based representations improve the accuracy of stock movement prediction models using deep learning techniques?

---

## ✅ Stage 1: COMPLETED - Comprehensive Data Collection

### Pillar 1: Sentiment Data ✅
- **Historical Coverage**: 2010-2025 systematic sentiment collection
- **Sources**: News, social media, financial platforms, Wayback News archives  
- **Quality**: Unbiased historical intervals using AI sentiment analysis
- **Volume**: ~1,300-1,600 sentiment records with behavioral pattern analysis
- **Files**: `tesla_sentiment_complete_*.csv`, `tesla_social_*.csv`

### Pillar 2: Market Data (Macro/Technical/Fundamental) ✅
- **Technical Analysis**: 55+ indicators (RSI, MACD, Bollinger Bands, etc.)
- **Macro Economics**: 21+ indicators (VIX, SPY, Treasury yields, FRED data)
- **Fundamental Data**: Tesla financials, quarterly earnings, competitive analysis
- **OpenBB Integration**: Institutional-grade economic calendar, sector rotation
- **Files**: `tesla_comprehensive_*.csv`, `tesla_mega_dataset_auto_*.csv`, `openbb_tesla_competitive_analysis_*.csv`

### Data Infrastructure ✅
- **Mega-Dataset**: Unified timeseries with 200+ features
- **Quality**: Data completeness >95%, systematic validation
- **Period**: April 2024 - March 2025 (as per methodology)

---

## 🚀 Stage 2: IN PROGRESS - FinNeX Framework Implementation

### Phase 3: Mega-Image Construction (Week 4)
**Objective**: Transform multimodal data into structured 512x512 images

#### Image Structure Design:
```
┌─────────────────────────────────────┐
│           TOP SECTION               │
│     Macroeconomic Indicators        │
│  (VIX, Rates, Sector Rotation)      │
├─────────────────────────────────────┤
│          MIDDLE SECTION             │
│   Price/Volume + Sentiment          │
│     (OHLCV + Sentiment Heatmaps)    │
├─────────────────────────────────────┤
│          BOTTOM SECTION             │
│    Fundamentals + Competitive       │
│   (Ratios, P/E, Competitive Pos)    │
└─────────────────────────────────────┘
```

#### Technical Implementation:
- **Encoding Strategy**: Normalized values → distinct color mappings in 2x2 pixel blocks
- **Data Sources**: Combine Pillar 1 + Pillar 2 datasets
- **Output**: 512x512 Mega-Images for each trading day
- **Tools**: Python (NumPy, PIL, matplotlib), custom normalization functions

### Phase 4: Patch Dataset Preparation (Week 5)
**Objective**: Prepare CNN training data from Mega-Images

#### Patch Processing:
- **Patch Size**: 32x32 pixels
- **Grid Structure**: 16x16 grid per Mega-Image  
- **Labels**: Buy/Hold/Sell signals based on next-day price movement
- **Data Augmentation**: Random flips, brightness/contrast jitter, Gaussian noise
- **Train/Test Split**: Stratified split maintaining temporal order

### Phase 5: CNN Model Development (Weeks 6-7)
**Objective**: Train CNN classifier for stock signal prediction

#### Model Architecture:
- **Base CNN**: 2 convolutional blocks + max-pooling + ReLU + FC layers
- **Input**: 32x32x3 patches from Mega-Images
- **Output**: 3-class classification (Buy/Hold/Sell)
- **Regularization**: Dropout, early stopping, L2 regularization
- **Framework**: PyTorch/TensorFlow

#### Evaluation Metrics:
- Accuracy, Precision, Recall, F1-Score
- Confusion matrices across signal categories
- Cross-validation if dataset size allows

### Phase 6: Hybrid CNN-Transformer Extension (Weeks 8-10)
**Objective**: Explore advanced architectures for enhanced performance

#### Advanced Models:
- **Vision Transformer**: Process patch embeddings
- **Hybrid CNN-ViT**: Capture local (CNN) + global (Transformer) features
- **Attention Mechanisms**: Cross-modal attention between price and sentiment
- **Research Contribution**: Address unified multimodal framework gap

---

## 📊 Expected Contributions

### Technical Deliverables:
1. **FinNeX Framework**: Novel multimodal image construction method
2. **Predictive Prototype**: CNN-based Buy/Hold/Sell classifier
3. **Academic Paper**: Evaluation of multimodal fusion effectiveness
4. **Code Repository**: Reproducible implementation

### Research Contributions:
1. **Multimodal Integration**: Address literature gap in unified frameworks
2. **Behavioral Finance**: Capture sentiment-driven market dynamics  
3. **Visual Encoding**: Novel approach to financial data representation
4. **Academic Impact**: Bridge computational finance and deep learning

---

## 🔧 Technical Resources

### Hardware Requirements:
- **GPU**: Azure Cloud VM (if local resources insufficient)
- **RAM**: 16GB+ for large dataset processing
- **Storage**: ~50GB for images and model checkpoints

### Software Stack:
- **Python**: PyTorch/TensorFlow, NumPy, pandas, PIL
- **Data Sources**: Already collected (Pillar 1 + Pillar 2)
- **Visualization**: matplotlib, seaborn for analysis
- **NLP**: FinBERT (already integrated in Pillar 1)

### Libraries:
```python
# Core ML
import torch
import torchvision
from sklearn.metrics import classification_report

# Data Processing  
import pandas as pd
import numpy as np
from PIL import Image

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## 📈 Success Metrics

### Quantitative Targets:
- **Baseline Accuracy**: >70% (vs random 33%)
- **Literature Comparison**: Outperform Kusuma et al. (92.2%) with multimodal data
- **Hybrid Model**: >15% improvement over CNN-only (Zhang et al. benchmark)

### Qualitative Goals:
- **Interpretability**: Visual feature explanation capability
- **Scalability**: Framework applicable to other stocks/markets  
- **Academic Quality**: Publication-ready methodology and results

---

## ⚠️ Risk Mitigation

### Identified Risks:
1. **Model Overfitting**: Apply regularization, cross-validation
2. **Computational Limits**: Use Azure GPU VMs, optimize batch sizes
3. **Data Noise**: Robust preprocessing, outlier detection
4. **Time Constraints**: Weekly milestones, supervisor check-ins

### Contingency Plans:
- **Simplified Models**: CNN-only if hybrid models too complex
- **Reduced Scope**: Focus on Buy/Sell (binary) vs 3-class classification
- **Alternative Validation**: Hold-out testing if cross-validation impractical

---

## 📅 Timeline Summary

| Phase | Duration | Key Deliverable | Status |
|-------|----------|----------------|---------|
| 1-2 | Weeks 1-3 | Data Collection | ✅ COMPLETE |
| 3 | Week 4 | Mega-Image Generator | 🎯 NEXT |
| 4 | Week 5 | Patch Datasets | ⏳ PENDING |
| 5 | Weeks 6-7 | CNN Training | ⏳ PENDING |
| 6 | Weeks 8-10 | Hybrid Models + Paper | ⏳ PENDING |

---

## 📚 Key References

### Literature Foundation:
- **Sezer & Ozbayoglu (2018)**: GAF-based CNN methodology
- **Kusuma et al. (2019)**: Candlestick CNN baseline (92.2% accuracy)
- **AI et al. (2025)**: Multimodal transformer approach (23.7% RMSE reduction)

### Research Gaps Addressed:
1. **Unified Multimodal Framework**: Current methods process data separately
2. **Visual Encoding Integration**: Lack of systematic sentiment→image conversion
3. **Behavioral + Technical Fusion**: Missing comprehensive market context

---

## 🎯 Next Immediate Actions

### Week 4 Priority Tasks:
1. **Design Mega-Image Structure**: Define pixel layouts for each data type
2. **Implement Normalization**: Create robust data→image conversion functions  
3. **Generate Sample Images**: Create visualization pipeline for validation
4. **Data Integration**: Merge Pillar 1 + Pillar 2 datasets by trading dates

### Code Development:
```python
# Priority functions to implement:
def create_mega_image(price_data, sentiment_data, macro_data, fundamental_data)
def normalize_to_pixels(data_array, pixel_range=(0, 255))  
def generate_image_dataset(merged_dataframe, output_dir)
def create_patch_labels(price_series, threshold=0.02)
```

---

*This roadmap serves as the definitive guide for completing the FinNeX MSc project, ensuring systematic progress toward successful academic contribution in multimodal financial ML.*
