# Winnow — Vercel demo (single-page classifier)

A cut-down deployment of the spam classifier for Vercel: paste an email, get a verdict and an
explanation. **This is a demo, not the full application.** It has no authentication, dashboard,
batch scoring, model-evaluation suite, or admin page — those live in the Streamlit app in the
parent directory, which needs a host that supports long-running processes.

## Why the model is exported instead of loaded

Vercel runs serverless functions with a bundle size limit and a cold start on every idle request.
Importing scikit-learn would add roughly 165 MB to the bundle and seconds to each cold start, to
compute what is ultimately a sparse dot product.

So `build_model.py` trains the pipeline and exports its vocabulary, IDF weights, and coefficients
to `api/model.json` (0.6 MB). `api/scorer.py` reimplements the scoring path — sublinear TF,
smoothed IDF, L2 normalisation, 1–2 grams, English stop words, then a logistic sigmoid — using
the Python standard library alone. `requirements.txt` is intentionally empty.

`build_model.py` refuses to write the export unless the pure-Python scorer reproduces
scikit-learn's probabilities to within 1e-6 on held-out emails. The current export agrees to
**1.1e-7**, the residual being the 6-decimal rounding applied to coefficients.

## Model

| Metric | Value |
|---|---|
| Accuracy | 99.05% |
| Precision | 98.57% |
| Recall | 99.45% |
| F1 | 0.9901 |
| ROC AUC | 0.9992 |

Logistic regression over 20,000 TF-IDF features, measured on 6,093 held-out emails. The full
application selects a calibrated Linear SVC instead (99.21%); logistic regression is used here
because a single coefficient vector and a plain sigmoid export cleanly, and the 0.16 point
difference does not justify the extra complexity in a demo.

## Deploying

The repository root holds the Streamlit app, so point Vercel at this subdirectory:

1. Import the GitHub repository at [vercel.com/new](https://vercel.com/new).
2. Set **Root Directory** to `vercel-demo`.
3. Leave the framework preset as **Other**. Do not set a build command.
4. Deploy.

No environment variables are needed — the demo makes no external API calls.

## Rebuilding the model

From the repository root, after retraining:

```bash
python vercel-demo/build_model.py
```

This rewrites `api/model.json` and re-runs the parity check against scikit-learn.

## Layout

```
vercel-demo/
├── index.html          # Single-page frontend, no build step
├── api/
│   ├── classify.py     # Serverless function: GET metadata, POST classify
│   ├── scorer.py       # Pure-Python TF-IDF + logistic scoring
│   └── model.json      # Exported vocabulary, IDF weights, coefficients
├── build_model.py      # Trains and exports the model, verifies parity
├── requirements.txt    # Intentionally empty
└── vercel.json         # Function memory, timeout, bundled files
```

## API

```bash
curl -X POST https://<your-deployment>/api/classify \
  -H 'Content-Type: application/json' \
  -d '{"subject":"You have won","body":"Click here to claim your prize"}'
```

```json
{
  "label": "spam",
  "spam_probability": 0.9772,
  "confidence": 0.9544,
  "decision": "quarantine",
  "threshold": 0.5,
  "top_terms": [{ "term": "claim", "contribution": 0.412, "direction": "spam" }]
}
```

`GET /api/classify` returns the model's metrics instead, which is what the page uses to render
its accuracy figures.
