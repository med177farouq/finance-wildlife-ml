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
