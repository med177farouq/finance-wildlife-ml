One-Month Plan (March 29 - April 28, 2025)

With 30 days, we’ll split time across the literature review, project plan, and presentation. Here’s a draft schedule—adjustable after brainstorming:
Week 1 (March 29 - April 4): Literature Review Kickoff

    Goal: Gather and skim 15-20 key papers on financial ML, CNNs, and data-to-image methods.
    Tasks:
        March 29: Brainstorm, finalize concept, list keywords (e.g., “financial time-series CNN,” “multi-modal finance prediction”).
        March 30-31: Search arXiv, Google Scholar, Springer for recent papers (2020-2025). I’ll help dig up sources.
        April 1-4: Read abstracts, sort into buckets (e.g., price prediction, image methods, hybrid models), note gaps.
    Output: A list of papers with 1-2 sentence summaries each.

Week 2 (April 5-11): Deep Dive and Drafting

    Goal: Write the literature review’s core sections.
    Tasks:
        April 5-7: Read 5-7 key papers in depth (e.g., Sezer & Ozbayoglu 2018, Hoseinzade 2019). I’ll summarize tricky bits.
        April 8-10: Draft intro (why this matters), methods (what’s been done), and gaps (your niche). Aim for 1,500-2,000 words.
        April 11: Refine based on feedback (I’ll review).
    Output: Rough draft of lit review.

Week 3 (April 12-18): Project Plan and Polish

    Goal: Finalize lit review, draft project plan.
    Tasks:
        April 12-14: Finish lit review (add future directions, e.g., “multi-modal image fusion is underexplored”).
        April 15-17: Draft project plan:
            Scope: Predict price/risk for 1-2 stocks.
            Data: Prices (Yahoo), news (NewsAPI), sentiment (X).
            Steps: Collect, convert to images, train CNN, test, deploy.
            Timeline: April-Aug (data prep in May, coding in June, testing in July, finalizing in Aug).
        April 18: Sync with me to tweak.
    Output: Final lit review (~2,500 words), project plan (~1,000 words).

Week 4 (April 19-28): Presentation and Buffer

    Goal: Create a compelling slide deck, buffer for polish.
    Tasks:
        April 19-21: Build slides:
            Intro: Problem (finance noise), your twist (images + CNNs).
            Lit Review: Key findings, your gap.
            Plan: Steps, timeline, deliverables.
            Impact: Trading tool + academic paper.
        April 22-24: Practice run-through (I’ll critique timing, clarity).
        April 25-27: Final edits to all deliverables.
        April 28: Submit or present!
    Output: 10-15 slide deck, polished docs.

Full Project Timeline (April 29 - August 31, 2025)

Post-April, here’s a high-level plan to finish by August:

    May: Collect data (e.g., Tesla prices, news, X posts), prototype image conversion.
    June: Code CNN in Azure ML, train on small dataset.
    July: Test, tweak (e.g., add data types), simulate trading profits.
    August: Deploy model, draft paper, finalize results.
# Project Scope - April 5, 2025
- **Concept**: Frame Tesla signals as "wildlife behaviors" (roaring = 2%-3% spikes, purring = stability, hissing = volatility).
- **Data**: Tesla prices (Yahoo Finance), X sentiment, news (NewsAPI) for April 2024 - March 2025.
- **Split**: Technical (prices), Qualitative (news), Behavioral (X sentiment).
- **Images**: Stacked layers (candlestick + sentiment + news heatmap) for 5-day windows.
- **Target**: Price direction (Up/Down) + Spike type (Temporary/Permanent).
- **Goals**: Practical tool for Tesla prediction; academic paper on multi-modal image fusion.

# Week 2 Plan (Adjusted for Your Schedule)

    Goal: Deep dive into the 7 must-read papers, draft the lit review.
    Tasks:
        April 6 (Sunday): I’ll post paper summaries by 2 PM GMT+1. You review anytime after.
        April 7 (Monday): You skim the summaries, note any questions. We can chat at 2 PM GMT+1 if you’re ready.
        April 8-10 (Tuesday-Thursday): You’re in classes 10 AM - 5 PM, so we’ll connect at 6 PM GMT+1 each day:
            April 8: I’ll provide a lit review structure (intro, methods, gaps). You start drafting in lit_review.md.
            April 9: Continue drafting—we’ll discuss progress.
            April 10: Finish rough draft (~1,500 words).
        April 11 (Friday): Review and refine at 2 PM GMT+1.
    Output: Rough draft of lit review (~1,500 words)
# Project Scope - April 6, 2025
- **Concept**: Tesla signals as "wildlife behaviors" (roaring = 1%-3% spikes, purring = stability, hissing = volatility).
- **Data**: Tesla prices, X sentiment, news (NewsAPI) for April 2024 - March 2025.
- **Split**: Technical (prices), Qualitative (news: Insider vs. Outsider factors), Behavioral (X sentiment).
- **Images**: Stacked layers (candlestick + sentiment + news heatmap, non-overlapping 5-day windows).
- **Target**: Price direction (Up/Down → Buy/Sell) + Spike type (Temporary/Permanent).
- **Goals**: Practical tool for Tesla prediction; academic paper on multi-modal image fusion.
