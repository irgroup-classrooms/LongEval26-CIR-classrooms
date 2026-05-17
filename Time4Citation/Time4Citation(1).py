import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prototyp: Multi-Stage Retrieval Pipeline für die LongEval-Challenge

    Dieses System implementiert eine robuste Retrieval-Architektur zur Verarbeitung der LongEval-Sci-Kollektion. Der Fokus liegt auf einer modularen Pipeline, die klassische probabilistische Modelle (BM25) mit modernem Feature Engineering (zeitbasierte Dynamik und Zitationsmetriken) kombiniert.

    **Kernkomponenten:**
    - **Core Engine:** PyTerrier Framework für effizientes Indexing und Retrieval.
    - **Data Pipeline:** Automatisierte Ingestion via `ir_datasets`.
    - **Advanced Scoring:** Implementierung von Conditional Recency Boosting und Citation-Impact-Gewichtung.
    - **Validation:** Systematisches Benchmarking mittels NDCG und MAP zur Performance-Optimierung.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Data Ingestion & Konfiguration
    Initialisierung der LongEval-Integration zur Gewährleistung der Datenintegrität und Reproduzierbarkeit über verschiedene Snapshots hinweg.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Dependency Management:** Laden der erforderlichen Bibliotheken für High-Performance Computing und Information Retrieval.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Framework Initialisierung:** Start der PyTerrier JVM-Umgebung.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Dataset Loading:** Abruf des *train*-Snapshots (`2024-11`). Diese automatisierte Ingestion stellt sicher, dass die Pipeline auf validierten Datenquellen aufsetzt.
    """)
    return


@app.cell
def _():
    import pyterrier as pt
    from ir_datasets_longeval import load
    import pandas as pd
    import polars as pl
    from datetime import datetime
    import numpy as np
    import requests
    from time import sleep
    from pathlib import Path
    import re
    import time
    import urllib.parse
    from difflib import SequenceMatcher
    from tqdm import tqdm
    import json
    import matplotlib.pyplot as plt

    return datetime, load, np, pd, pl, plt, pt, re


@app.cell
def _(load):
    dataset = load("longeval-sci-2026/snapshot-1/train/raw")
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Inverted Index Construction
    Umwandlung der Rohdaten (TREC-Format) in einen effizienten Inverted Index. Wir nutzen den `IterDictIndexer`, um Metadatenfelder wie `year` und `text` direkt im Index für nachgelagerte Scoring-Phasen verfügbar zu machen.
    """)
    return


@app.cell
def _(pt):
    indexer = pt.IterDictIndexer(
        index_path="C:/Projektdings/dis18-2025/tutorials/index", # Lokaler Pfad zur Index-Persistenz
        overwrite=True, 
        meta={"docno": 100, "text": 20480, "year": 4}  # Schema-Definition für Metadaten
    )
    return (indexer,)


@app.cell
def _(dataset, datetime):
    # Dokument-Vorverarbeitung und Feature-Extraktion (Temporal Metadata)
    def document_generator():
        for doc in dataset.docs_iter():
            # Fallback-Logik für Datumsmetadaten zur Sicherstellung der Datenvollständigkeit
            date_str = doc.publishedDate or doc.createdDate
            year = datetime.fromisoformat(date_str).year
            yield {
                "docno": doc.doc_id,    
                "text": doc.default_text(),
                "year": str(year),
            }

    docs = document_generator()
    return (docs,)


@app.cell
def _(dataset):
    dataset.has_qrels()
    return


@app.cell
def _(docs, indexer):
    indexref = indexer.index(docs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3. Retrieval Engine & Baseline
    Nach dem Indexing (ca. 2 Mio. Dokumente) wird die Retrieval Engine initialisiert. Als Baseline dient ein optimiertes BM25-Modell.
    """)
    return


@app.cell
def _(pt):
    index = pt.IndexFactory.of("C:/Projektdings/dis18-2025/tutorials/index")
    return (index,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Validierung der Index-Integrität: Erwartete Dokumentenanzahl 2.014.265.
    """)
    return


@app.cell
def _(index, pt):
    # Initialisierung der Retrieval-Modelle mit Metadaten-Support
    bm25 = pt.terrier.Retriever(index, wmodel="BM25", metadata=["year", "docno"])
    tf = pt.terrier.Retriever(index, wmodel="Tf", metadata=["year", "docno"])
    return (bm25,)


@app.cell
def _(bm25, topics):
    bm25(topics)
    return


@app.cell
def _(bm25):
    print(bm25.search("peer-to-peer communication").head(4))
    return


@app.cell
def _(bm25, dataset, pd, re):
    # Pre-Processing der Queries zur Optimierung der Retrieval-Qualität
    topics = pd.DataFrame(dataset.queries)
    topics.rename(columns={"query_id": "qid", "text": "query"}, inplace=True)

    def terrier_safe_query(q):
        if pd.isna(q):
            return ""
        q = str(q)
        # Bereinigung von Sonderzeichen zur Vermeidung von Parser-Fehlern
        q = re.sub(r"[^A-Za-z0-9\s]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    topics["query"] = topics["query"].apply(terrier_safe_query)

    # Analyse des Candidate Retrieval Sets
    cands = bm25.transform(topics)
    top50_cands = cands[cands["rank"] < 50]
    unique_docnos = set(top50_cands["docno"].apply(lambda x: str(int(float(x)))))
    print(f"Unique top-50 docnos across {len(topics)} queries: {len(unique_docnos)}")
    return terrier_safe_query, topics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. Feature Engineering: Conditional Recency Boosting
    In diesem Abschnitt implementieren wir eine zeitbewusste Re-Ranking-Logik. Dokumente werden basierend auf ihrem Alter und der durchschnittlichen Aktualität des Top-K Ergebnissatzes gewichtet.
    """)
    return


@app.cell
def _(dataset, datetime, pd):
    # Berechnung der Korpus-Statistiken für die zeitliche Normierung
    rows = []
    for doc in dataset.docs_iter():
        date_str = doc.publishedDate or doc.createdDate
        year = int(datetime.fromisoformat(date_str).year)
        rows.append({"docno": doc.doc_id, "year": year})

    doc_years = pd.DataFrame(rows)
    CORPUS_MEAN_YEAR = doc_years["year"].mean()

    print(CORPUS_MEAN_YEAR)

    doc_years["year"].median()
    return CORPUS_MEAN_YEAR, doc_years


@app.cell
def _(doc_years, plt):
    # Explorative Analyse der zeitlichen Verteilung im Korpus
    year_counts = doc_years["year"].value_counts().sort_index()

    plt.bar(year_counts.index, year_counts.values)
    plt.xlabel("Year")
    plt.ylabel("Number of Documents")
    plt.title("Temporal Distribution of Corpus")
    plt.xlim(left=1900, right=2026)
    plt.ylim(top=20000)
    plt.show()
    return


@app.cell
def _(pd, pt):
    # Funktion zur Aggregation der zeitlichen Metadaten pro Query-Result-Set
    K = 50 

    def add_avg_year_topk(df_q):
        df_q = df_q.sort_values("rank")
        topk = df_q.head(K)
        years = pd.to_numeric(topk["year"], errors="coerce")
        mean_year = years.mean()
        df_q["avg_year_topk"] = mean_year
        return df_q

    avg_year_t = pt.apply.by_query(add_avg_year_topk)
    return (avg_year_t,)


@app.cell
def _(CORPUS_MEAN_YEAR, avg_year_t, bm25, np, pt):
    # Implementierung der exponentiellen Decay-Funktion für Recency Boosting
    CURRENT_YEAR = 2025
    ALPHA = 4.0
    DELTA = 1.0 

    def conditional_recency_boost(row):
        try:
            year = int(row["year"])
            avg_topk = float(row["avg_year_topk"])
        except (TypeError, ValueError):
            return row["score"]

        # Logik: Wenn die Top-K Ergebnisse überdurchschnittlich aktuell sind, wird Recency höher gewichtet
        if avg_topk > CORPUS_MEAN_YEAR + DELTA:
            age = max(0, CURRENT_YEAR - year)
            recency = np.exp(-0.3 * age)
            return row["score"] + 2* ALPHA * recency
        elif avg_topk > CORPUS_MEAN_YEAR:
            age = max(0, CURRENT_YEAR - year)
            recency = np.exp(-0.3 * age)
            return row["score"] + ALPHA * recency
        else:
            return row["score"]
    

    # Pipeline-Definition: BM25 -> Metadaten-Aggregation -> Recency-Scoring
    bm25_with_avg = bm25 >> avg_year_t
    bm25_cond_recency = bm25_with_avg >> pt.apply.doc_score(conditional_recency_boost)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5. Integration von Impact-Metriken (Citation Boosting)
    Wissenschaftliche Relevanz wird hier durch die Integration von Zitationsdaten (Open Citations) als externes Qualitätssignal abgebildet.
    """)
    return


@app.cell
def _(pl):
    # Analyse und Pre-Processing der Zitationsdaten (Snapshot-basiert)
    CUTOFF = pl.date(2026,1,1)

    citations = pl.read_csv("open_citations.csv")

    citations = citations.with_columns(
        pl.col("creation").str.strptime(pl.Date, strict=False)
    )

    # Aggregation der Zitationen pro Dokument
    citation_stats = citations.group_by("cited_doc_id").agg([
        pl.len().alias("citations"),
        pl.col("creation").max().alias("latest_citation_date")
    ])

    citation_stats = citation_stats.rename({"cited_doc_id": "docno"})
    return citation_stats, citations


@app.cell
def _(citations):
    citations
    return


@app.cell
def _(citations, pl, plt):
    # Statistische Auswertung der Zitations-Latenz
    from datetime import date
    citations_1 = citations.with_columns(((pl.lit(date.today()) - pl.col('creation')).dt.total_days() / 365).alias('age_years'))
    ages = citations_1['age_years'].to_numpy()
    plt.hist(ages, bins=100)
    plt.xlim(left=0, right=40)
    plt.xlabel('Citation Age (Years)')
    plt.ylabel('Frequency')
    plt.title('Citation Age Distribution')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6. Daten-Bereinigung und Merging
    Vorbereitung der Zitationsdaten für die Integration in die PyTerrier-Pipeline.
    """)
    return


@app.cell
def _(citation_stats, pd):
    orig_df = citation_stats.to_pandas()
    tmp = orig_df.copy()
    tmp["docno"] = pd.to_numeric(tmp["docno"], errors="coerce")

    num_total = len(tmp)
    num_invalid = tmp["docno"].isna().sum()

    print(f"Total rows: {num_total}")
    print(f"Data Quality: {(1 - num_invalid / num_total):.4%} valid docnos")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7. Modell-Varianten: Citation Boosting
    Implementierung verschiedener Strategien: Logarithmisches Boosting, binäres Boosting und zeitlich gewichtetes Zitations-Scoring.
    """)
    return


@app.cell
def _(bm25, citation_stats, np, pd, pt):
    # Modell 1: Logarithmisches Citation-Scaling
    citation_df = citation_stats.to_pandas()
    citation_df["docno"] = pd.to_numeric(citation_df["docno"], errors="coerce")
    citation_df = citation_df.dropna(subset=["docno"])
    citation_df["docno"] = citation_df["docno"].astype(int).astype(str)

    def add_citations(df):
        df = df.copy()
        df["docno"] = df["docno"].astype(str)
        df = df.merge(citation_df, on="docno", how="left")
        df["citations"] = df["citations"].fillna(0).astype(int)
        return df

    add_citations_t = pt.apply.generic(add_citations)

    def citations_boost(row):
        c = row["citations"]
        return row["score"] + 0.3 * np.log1p(c)

    bm25_cit = bm25 >> add_citations_t >> pt.apply.doc_score(citations_boost)
    return (add_citations_t,)


@app.cell
def _(add_citations_t, bm25):
    _res = (bm25 >> add_citations_t).search('corticosteroid')
    print(_res[['docno', 'score', 'citations', 'latest_citation_date']].head(10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Binäres Impact-Feature:** Klassifizierung von Dokumenten in "zitiert" vs "nicht zitiert".
    """)
    return


@app.cell
def _(add_citations_t, bm25, pt):
    def _citations_class_boost(row):
        c = float(row.get('citations', 0))
        if c >= 1:
            c = 1
        return row['score'] + c
    bm25_cit_binary = bm25 >> add_citations_t >> pt.apply.doc_score(_citations_class_boost)
    return (bm25_cit_binary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Temporal Citation Decay:** Kombination aus Aktualität der Zitation und Retrieval-Score.
    """)
    return


@app.cell
def _(add_citations_t, bm25, datetime, pd, pt):
    def citations_recency_class_boost(row):
        score = row['score']
        date = row.get('latest_citation_date', None)
        if pd.isna(date):
            return score
        days = (datetime.now() - date).days
        if days <= 365 * 9: # Fokus auf neuere wissenschaftliche Trends
            boost = 1.5 
        else:
            boost = 0.5
        return score * (1 + boost)
    bm25_cit_recency = bm25 >> add_citations_t >> pt.apply.doc_score(citations_recency_class_boost)
    return (bm25_cit_recency,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 8. Quantitative Evaluation (Benchmarking)
    Systematischer Vergleich der entwickelten Retrieval-Strategien gegenüber der Baseline unter Verwendung offizieller Metriken.
    """)
    return


@app.cell
def _(dataset, pd, terrier_safe_query):
    topics_1 = pd.DataFrame(dataset.queries)
    topics_1.rename(columns={'query_id': 'qid', 'text': 'query'}, inplace=True)
    topics_1['query'] = topics_1['query'].apply(terrier_safe_query)
    return


@app.cell
def _(
    bm25,
    bm25_cit_binary,
    bm25_cit_recency,
    dataset,
    pd,
    pt,
    terrier_safe_query,
):
    # Durchführung des Experiments mit statistischer Signifikanzprüfung
    topics_2 = pd.DataFrame(dataset.queries)
    topics_2.rename(columns={'query_id': 'qid', 'text': 'query'}, inplace=True)
    topics_2['query'] = topics_2['query'].apply(terrier_safe_query)
    qrels = pd.DataFrame(dataset.qrels)
    qrels.rename(columns={'query_id': 'qid', 'doc_id': 'docno', 'relevance': 'label'}, inplace=True)
    _results = pt.Experiment(
        retr_systems=[bm25, bm25_cit_binary, bm25_cit_recency], 
        topics=topics_2, 
        qrels=qrels, 
        baseline=0, 
        correction='bonferroni', 
        eval_metrics=['ndcg', 'map', 'ndcg_cut_10'], 
        verbose=True, 
        save_dir='runs', 
        save_mode='overwrite', 
        names=['BM25_Baseline', 'Citation_Binary_Boost', 'Temporal_Impact_Boost']
    )
    print(_results)
    _results.to_csv('benchmarking_results.csv')
    return qrels, topics_2


@app.cell
def _(add_citations_t, bm25, pt, qrels, topics_2):
    # Hyperparameter-Optimierung für den Binary Boost Faktor
    boost = 0.2
    while boost <= 2:
        def _citations_class_boost(row):
            c = float(row.get('citations', 0))
            if c >= 1:
                c = boost
            return row['score'] + c
        bm25_cit_binary_1 = bm25 >> add_citations_t >> pt.apply.doc_score(_citations_class_boost)
        _results = pt.Experiment(retr_systems=[bm25, bm25_cit_binary_1], topics=topics_2, qrels=qrels, eval_metrics=['ndcg', 'map'], names=['BM25', f'BM_CIT_bin_{boost}'])
        boost = boost + 0.2
    return


@app.cell
def _(bm25, pt, topics_2):
    # Generierung des finalen Run-Files für das Leaderboard-Submission
    _res = bm25(topics_2)
    pt.io.write_results(_res, 'FINAL_SUBMISSION_RUN.txt')
    return


@app.cell
def _(qrels, topics_2):
    # Statistische Verteilung der Relevanz-Labels im Test-Set
    print(qrels['label'].value_counts())
    print('Unique queries in Qrels:', qrels['qid'].nunique())
    return


if __name__ == "__main__":
    app.run()