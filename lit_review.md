# Literature Review - Week 1 (March 29 - April 4, 2025)

## Initial Search Results
Searched arXiv, Google Scholar, Springer for papers on financial CNNs, multi-modal prediction, text analytics in finance. Found 15 relevant papers (2020-2025):

1. Sezer & Ozbayoglu (2020) - Converted price series to images (GAF, MTF), used CNNs for trading.
2. Hoseinzade & Haratizadeh (2019) - Used candlestick charts for CNN stock prediction.
3. Mohan et al. (2019) - Combined news sentiment with prices, no images.
4. Chen et al. (2018) - Fused prices and news, used LSTM.
5. Sirignano & Cont (2019) - CNNs on order book heatmaps.
6. Cao & Wang (2019) - CNNs for price forecasting.
7. Chatzis et al. (2018) - CNNs for crisis prediction.
8. Peng & Yan (2021) - Deep learning for credit risk.
9. Zhang et al. (2023) - Reviewed deep learning in finance.
10. Li et al. (2022) - Text analytics in finance survey.
11. Kim & Lee (2020) - Twitter sentiment for stock prediction.
12. Wang et al. (2021) - Multi-modal forecasting, no images.
13. Barra et al. (2020) - Volatility forecasting with RNNs.
14. Jiang (2021) - Deep learning in stock prediction review.
15. Xu et al. (2024) - Visualizing time-series for CNNs.

## Must-Reads for Week 2
- Sezer & Ozbayoglu (2020)
- Hoseinzade & Haratizadeh (2019)
- Mohan et al. (2019)
- Zhang et al. (2023)
- Xu et al. (2024)
- Li et al. (2022)
- Barra et al. (2020)

## Paper Summaries - April 6, 2025
1. **Sezer & Ozbayoglu (2020)**: Converted price series to images (GAF, MTF), used CNNs for trading. Relevance: Validates image approach, but lacks sentiment/news.
2. **Hoseinzade & Haratizadeh (2019)**: Used candlestick charts for CNN prediction. Relevance: Aligns with our candlestick layer, we add more data.
3. **Mohan et al. (2019)**: Combined news sentiment with prices using LSTM. Relevance: Supports our data mix, but we use images.
4. **Zhang et al. (2023)**: Reviewed deep learning in finance, noted image methods as niche. Relevance: Sets context, shows our gap.
5. **Xu et al. (2024)**: Visualized time-series with GAF for CNNs. Relevance: Reinforces image methods, we extend with sentiment/news.
6. **Li et al. (2022)**: Surveyed text mining in finance. Relevance: Supports our news/sentiment use, we innovate with images.
7. **Barra et al. (2020)**: Used RNNs for volatility forecasting. Relevance: Supports our spike target, we use CNNs and images.

## Paper Summaries - April 6, 2025 (Updated with Comments)
1. **Sezer & Ozbayoglu (2020)**: Converted price series to images (GAF, MTF), used CNNs for trading. **Relevance**: Validates image approach (e.g., candlesticks, time-series@farouq), but lacks sentiment/news.
2. **Hoseinzade & Haratizadeh (2019)**: Used candlestick charts for CNN prediction. **Relevance**: Aligns with our candlestick layer, we add more data.
3. **Mohan et al. (2019)**: Combined news sentiment with prices using LSTM. **Relevance**: Supports our data mix, but we use images. We’ll filter news into Insider (fundamental) vs. Outsider (sentimental) factors using NLP tools (@farouq).
4. **Zhang et al. (2023)**: Reviewed deep learning in finance, noted image methods as niche. **Relevance**: Sets context, shows our gap. We aim for: 1) Classification (Up/Down, Buy/Sell); 2) Detection (1%-3% spikes); 3) Tracking (avoid double counting) (@farouq).
5. **Xu et al. (2024)**: Visualized time-series with GAF for CNNs. **Relevance**: Reinforces image methods, we extend with sentiment/news.
6. **Li et al. (2022)**: Surveyed text mining in finance. **Relevance**: Supports our news/sentiment use, we innovate with images.
7. **Barra et al. (2020)**: Used RNNs for volatility forecasting. **Relevance**: Supports our spike target, we use CNNs and images. Future: Add VIX/fear index to track spikes (@farouq).
