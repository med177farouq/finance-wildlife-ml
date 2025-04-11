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

---------------------------------------------------------
# Mega-Image Model (Lion’s Voice): A Financial Glyphboard for Predictive Learning - April 11, 2025
- **Concept**: Financial signals as “wildlife behaviors” (Roaring, Purring, Hissing), encoded in a Mega-Image—a visual embedding of the market’s multidimensional state.
- **Symbolism**: The Lion’s Voice directs the Bear and Bull, positioning the model as a leader of interpretation.
- **Data**: Tesla prices, X sentiment, news (NewsAPI), fundamentals, macro indicators for April 2024 - March 2025 (test dataset; model generalizable to any stock).
- **Split**: Technical (prices, TS2Vec embeddings), Qualitative (news: FinBERT embeddings, clustered into topics), Behavioral (X sentiment with exponential decay), Fundamentals (dense embeddings), Macro Indicators (manual encoding).
- **Images**: Mega-Image (512×512 pixels, 2×2 pixel blocks = 256×256 grid):
  - Top (Rows 0-170): Rare forces (purple = inflation, dark gray = Fed policy, white/light blue/orange = M&A/IPOs/rates, purple = VIX).
  - Middle (Rows 171-340): Active market (green/red = price up/down, yellow/orange = sentiment, light green/light red = news).
  - Bottom (Rows 341-512): Fundamentals (blue = valuation, green = earnings, red = debt, yellow = dividends).
- **Methodology**:
  - Preprocess data with specialized embeddings (FinBERT, TS2Vec, dense embeddings).
  - Encode embeddings into the Mega-Image using color/position rules.
  - Slice the Mega-Image into 16×16 patches (16×16 = 256 patches total).
  - CNN to process the Mega-Image, extract features, and output state probabilities.
  - Markov Chain to model state transitions (Roaring, Purring, Hissing) with exponential convergence, using dynamic volatility thresholds.
  - Note: ViT or advanced techniques (e.g., hybrid CNN-ViT) are future options.
- **Target**:
  - Price direction (Buy/Sell/Hold).
  - Next 5-day path (Bounce/Stabilize/Continue).
  - Optional: Risk level (e.g., inferred from macro signals).
- **Goals**: Practical tool for stock prediction (tested on Tesla, generalizable); academic paper on visual data encoding, exponential convergence, and multi-modal fusion.
