# finance-wildlife-ml
# Project Scope - April 6, 2025
- **Concept**: Financial signals as "wildlife behaviors" (roaring = 1%-3% spikes, purring = stability, hissing = volatility).
- **Data**: Tesla prices, X sentiment, news (NewsAPI) for April 2024 - March 2025 (test dataset; model generalizable to any stock).
- **Split**: Technical (prices), Qualitative (news: Insider vs. Outsider factors), Behavioral (X sentiment).
- **Images**: Stacked layers (candlestick + sentiment + news heatmap, non-overlapping 5-day windows).
- **Target**: Price direction (Up/Down → Buy/Sell) + Spike type (Temporary/Permanent).
- **Goals**: Practical tool for stock prediction (tested on Tesla, generalizable); academic paper on multi-modal image fusion.
# Project Scope - April 8, 2025
- **Concept**: Financial signals as "wildlife behaviors" (roaring = 1%-3% spikes, purring = stability, hissing = volatility).
- **Data**: Tesla prices, X sentiment, news (NewsAPI) for April 2024 - March 2025 (test dataset; model generalizable to any stock).
- **Split**: Technical (prices), Qualitative (news: Insider vs. Outsider factors, filtered by importance), Behavioral (X sentiment).
- **Images**: Stacked layers (candlestick + sentiment + news heatmap, non-overlapping 5-day windows).
- **Methodology**:
  - CNN to extract features from images (filter noise, capture patterns).
  - Markov Chain to model state transitions (Roaring, Purring, Hissing) with exponential convergence for noisy data.
- **Target**: Price direction (Up/Down → Buy/Sell) + Spike type (Temporary/Permanent) + Expected return time to specific states.
- **Goals**: Practical tool for stock prediction (tested on Tesla, generalizable); academic paper on multi-modal image fusion and exponential convergence in Markov Chains.

# Mega Image Model (Lion’s Voice): A Unified Visual Representation of Financial Market States for Predictive Learning - April 11, 2025
- **Concept**: Financial signals as "wildlife behaviors" (Roaring, Purring, Hissing), encoded in a Mega-Image.
- **Data**: Tesla prices, X sentiment, news (NewsAPI), fundamentals, macro indicators for April 2024 - March 2025 (test dataset; model generalizable to any stock).
- **Split**: Technical (prices, TS2Vec embeddings), Qualitative (news: FinBERT embeddings, clustered into topics), Behavioral (X sentiment with exponential decay), Fundamentals (dense embeddings), Macro Indicators (manual encoding).
- **Images**: Mega-Image (512×512 pixels):
  - Background Layer: Fundamentals (market cap, valuation, sector).
  - Foreground Activity: Price (direction, volume, volatility), sentiment (intensity, news sentiment).
  - Sky & Environment: Macro shocks (inflation, Fed policy, M&A).
  - Top Banner Strip: Index/options signals (S&P 500, VIX, Put/Call Ratio).
- **Methodology**:
  - Preprocess data with specialized embeddings (FinBERT, TS2Vec, dense embeddings).
  - Encode embeddings into the Mega-Image using color/position rules.
  - ViT to process the Mega-Image, output state probabilities (hybrid model).
  - Markov Chain to model state transitions (Roaring, Purring, Hissing) with exponential convergence, using dynamic volatility thresholds.
- **Target**: Price direction (Buy/Sell/Hold) + Path prediction (Bounce/Stabilize/Continue) + Risk level.
- **Goals**: Practical tool for stock prediction (tested on Tesla, generalizable); academic paper on multi-modal image fusion, exponential convergence, and visual data encoding.

  # Mega Image Model (Lion’s Voice): A Financial Glyphboard for Predictive Learning - April 11, 2025
- **Concept**: Financial signals as "wildlife behaviors" (Roaring, Purring, Hissing), encoded in a Mega-Image—a visual embedding of the market’s multidimensional state.
- **Data**: Tesla prices, X sentiment, news (NewsAPI), fundamentals, macro indicators for April 2024 - March 2025 (test dataset; model generalizable to any stock).
- **Split**: Technical (prices, TS2Vec embeddings), Qualitative (news: FinBERT embeddings, clustered into topics), Behavioral (X sentiment with exponential decay), Fundamentals (dense embeddings), Macro Indicators (manual encoding).
- **Images**: Mega-Image (512×512 pixels, 2×2 pixel blocks):
  - Top (Rows 0-170): Macro signals/rare events (purple = inflation, dark gray = Fed policy, white/light blue/orange = M&A/IPOs/rates, purple = VIX).
  - Middle (Rows 171-340): Market flow/active layer (green/red = price up/down, yellow/orange = sentiment, light green/light red = news).
  - Bottom (Rows 341-512): Fundamentals/fixed layer (blue = valuation, green = earnings, red = debt, yellow = dividends).
- **Methodology**:
  - Preprocess data with specialized embeddings (FinBERT, TS2Vec, dense embeddings).
  - Encode embeddings into the Mega-Image using color/position rules.
  - Slice the Mega-Image into 16×16 patches.
  - ViT to process the Mega-Image, output state probabilities (hybrid model).
  - Markov Chain to model state transitions (Roaring, Purring, Hissing) with exponential convergence, using dynamic volatility thresholds.
- **Target**: Price direction (Buy/Sell) + Path prediction (Bounce/Stabilize/Continue) + Risk level.
- **Goals**: Practical tool for stock prediction (tested on Tesla, generalizable); academic paper on visual data encoding, exponential convergence, and multi-modal fusion.
