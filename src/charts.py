"""Altair chart builders shared across the dashboard pages."""

from __future__ import annotations

import altair as alt
import pandas as pd

SPAM_COLOR = "#e5484d"
HAM_COLOR = "#30a46c"
REVIEW_COLOR = "#f5a524"
ACCENT = "#5b7cfa"
MUTED = "#8b8f9e"

LABEL_SCALE = alt.Scale(domain=["ham", "spam"], range=[HAM_COLOR, SPAM_COLOR])
DECISION_SCALE = alt.Scale(
    domain=["deliver", "review", "quarantine"], range=[HAM_COLOR, REVIEW_COLOR, SPAM_COLOR]
)


def probability_bar(probability: float, threshold: float) -> alt.LayerChart:
    """Show one spam score against the decision threshold."""
    frame = pd.DataFrame({"metric": ["Spam score"], "value": [probability]})
    colour = SPAM_COLOR if probability >= threshold else HAM_COLOR

    track = (
        alt.Chart(pd.DataFrame({"metric": ["Spam score"], "value": [1.0]}))
        .mark_bar(color="#e3e5ea", cornerRadius=6, height=28)
        .encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title=None, axis=alt.Axis(format="%")),
            y=alt.Y("metric:N", title=None, axis=None),
        )
    )
    bar = (
        alt.Chart(frame)
        .mark_bar(color=colour, cornerRadius=6, height=28)
        .encode(
            x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("metric:N", title=None, axis=None),
            tooltip=[alt.Tooltip("value:Q", format=".2%", title="Spam probability")],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
        .mark_rule(color="#1f2430", strokeWidth=2, strokeDash=[4, 3])
        .encode(x="threshold:Q", tooltip=[alt.Tooltip("threshold:Q", format=".2f", title="Threshold")])
    )
    return (track + bar + rule).properties(height=60)


def confusion_heatmap(matrix: list[list[int]]) -> alt.LayerChart:
    """Render a confusion matrix as an annotated heatmap."""
    labels = ["ham", "spam"]
    rows = [
        {
            "Actual": labels[i],
            "Predicted": labels[j],
            "Count": matrix[i][j],
            "Outcome": "Correct" if i == j else "Error",
        }
        for i in range(2)
        for j in range(2)
    ]
    frame = pd.DataFrame(rows)

    base = alt.Chart(frame).encode(
        x=alt.X("Predicted:N", title="Predicted label", sort=labels),
        y=alt.Y("Actual:N", title="Actual label", sort=labels),
    )
    heat = base.mark_rect(cornerRadius=4, stroke="white", strokeWidth=3).encode(
        color=alt.Color(
            "Count:Q",
            scale=alt.Scale(scheme="blues"),
            legend=alt.Legend(title="Emails"),
        ),
        tooltip=["Actual", "Predicted", "Count", "Outcome"],
    )
    text = base.mark_text(fontSize=20, fontWeight="bold").encode(
        text=alt.Text("Count:Q", format=","),
        color=alt.condition(
            alt.datum.Count > frame["Count"].max() / 2, alt.value("white"), alt.value("#1f2430")
        ),
    )
    return (heat + text).properties(height=280)


def roc_chart(fpr: list[float], tpr: list[float], auc: float) -> alt.LayerChart:
    """Plot the ROC curve against the no-skill diagonal."""
    curve = pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr})
    diagonal = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

    line = (
        alt.Chart(curve)
        .mark_line(color=ACCENT, strokeWidth=2.5)
        .encode(
            x=alt.X("False positive rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("True positive rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            tooltip=[
                alt.Tooltip("False positive rate:Q", format=".3f"),
                alt.Tooltip("True positive rate:Q", format=".3f"),
            ],
        )
    )
    area = line.mark_area(color=ACCENT, opacity=0.12)
    baseline = (
        alt.Chart(diagonal)
        .mark_line(color=MUTED, strokeDash=[5, 4])
        .encode(x="x:Q", y="y:Q")
    )
    label = (
        alt.Chart(pd.DataFrame({"x": [0.62], "y": [0.22], "text": [f"AUC = {auc:.4f}"]}))
        .mark_text(fontSize=15, fontWeight="bold", color=ACCENT)
        .encode(x="x:Q", y="y:Q", text="text:N")
    )
    return (area + line + baseline + label).properties(height=320)


def pr_chart(precision: list[float], recall: list[float], average_precision: float) -> alt.LayerChart:
    """Plot the precision-recall curve."""
    frame = pd.DataFrame({"Precision": precision, "Recall": recall})
    line = (
        alt.Chart(frame)
        .mark_line(color=HAM_COLOR, strokeWidth=2.5)
        .encode(
            x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("Recall:Q", format=".3f"), alt.Tooltip("Precision:Q", format=".3f")],
        )
    )
    label = (
        alt.Chart(
            pd.DataFrame({"x": [0.3], "y": [0.35], "text": [f"Avg precision = {average_precision:.4f}"]})
        )
        .mark_text(fontSize=15, fontWeight="bold", color=HAM_COLOR)
        .encode(x="x:Q", y="y:Q", text="text:N")
    )
    return (line.mark_area(color=HAM_COLOR, opacity=0.12) + line + label).properties(height=320)


def threshold_chart(sweep: list[dict], current: float) -> alt.LayerChart:
    """Compare precision, recall, and F1 across candidate thresholds."""
    frame = pd.DataFrame(sweep).melt(
        id_vars="threshold",
        value_vars=["precision", "recall", "f1"],
        var_name="Metric",
        value_name="Score",
    )
    frame["Metric"] = frame["Metric"].str.replace("f1", "F1").str.title()

    lines = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=45))
        .encode(
            x=alt.X("threshold:Q", title="Decision threshold", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(range=[ACCENT, SPAM_COLOR, HAM_COLOR]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["Metric", alt.Tooltip("threshold:Q", format=".2f"), alt.Tooltip("Score:Q", format=".4f")],
        )
    )
    marker = (
        alt.Chart(pd.DataFrame({"threshold": [current]}))
        .mark_rule(color="#1f2430", strokeWidth=2, strokeDash=[4, 3])
        .encode(x="threshold:Q", tooltip=[alt.Tooltip("threshold:Q", title="In use", format=".2f")])
    )
    return (lines + marker).properties(height=320)


def cost_chart(sweep: list[dict], current: float) -> alt.LayerChart:
    """Show the weighted error cost across thresholds."""
    frame = pd.DataFrame(sweep)
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("threshold:Q", title="Decision threshold", bin=alt.Bin(step=0.05)),
            y=alt.Y("weighted_cost:Q", title="Weighted error cost"),
            color=alt.condition(
                abs(alt.datum.threshold - current) < 0.001, alt.value(ACCENT), alt.value("#c9cedb")
            ),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".2f"),
                "false_positives:Q",
                "false_negatives:Q",
                alt.Tooltip("weighted_cost:Q", format=".0f"),
            ],
        )
    )
    return (bars + bars.mark_bar(opacity=0)).properties(height=280)


def term_importance_chart(terms: list[dict], title: str, color: str) -> alt.Chart:
    """Rank the terms that push hardest toward one class."""
    frame = pd.DataFrame(terms)
    frame["magnitude"] = frame["weight"].abs()
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadius=3, color=color)
        .encode(
            x=alt.X("magnitude:Q", title="Model weight"),
            y=alt.Y("term:N", sort="-x", title=None),
            tooltip=["term", alt.Tooltip("weight:Q", format=".3f")],
        )
        .properties(height=max(200, 22 * len(frame)), title=title)
    )


def contribution_chart(contributions: list[dict]) -> alt.Chart:
    """Show which terms drove one individual prediction."""
    frame = pd.DataFrame(contributions)
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("contribution:Q", title="Contribution to spam score"),
            y=alt.Y("term:N", sort=alt.EncodingSortField(field="contribution", op="max"), title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["legitimate", "spam"], range=[HAM_COLOR, SPAM_COLOR]),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["term", alt.Tooltip("contribution:Q", format=".4f"), "direction"],
        )
        .properties(height=max(180, 26 * len(frame)))
    )


def class_balance_chart(counts: pd.Series) -> alt.Chart:
    """Show how many emails fall into each class."""
    frame = counts.reset_index()
    frame.columns = ["label", "count"]
    return (
        alt.Chart(frame)
        .mark_arc(innerRadius=70, cornerRadius=4)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color("label:N", scale=LABEL_SCALE, legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["label", alt.Tooltip("count:Q", format=",")],
        )
        .properties(height=280)
    )


def probability_distribution(frame: pd.DataFrame) -> alt.Chart:
    """Show how scores separate across the two true classes."""
    return (
        alt.Chart(frame)
        .mark_area(opacity=0.55, interpolate="step")
        .encode(
            x=alt.X(
                "spam_probability:Q",
                bin=alt.Bin(maxbins=40),
                title="Predicted spam probability",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y("count()", title="Emails", stack=None),
            color=alt.Color("true_label:N", scale=LABEL_SCALE, legend=alt.Legend(orient="top", title=None)),
            tooltip=["true_label", "count()"],
        )
        .properties(height=300)
    )


def feature_comparison_chart(frame: pd.DataFrame, feature: str) -> alt.Chart:
    """Compare one engineered feature's distribution between classes."""
    return (
        alt.Chart(frame)
        .mark_boxplot(size=45, extent="min-max")
        .encode(
            x=alt.X("label:N", title=None),
            y=alt.Y(f"{feature}:Q", title=feature.replace("_", " ").title()),
            color=alt.Color("label:N", scale=LABEL_SCALE, legend=None),
        )
        .properties(height=300)
    )


def timeline_chart(frame: pd.DataFrame) -> alt.Chart:
    """Plot classification volume over time, split by predicted label."""
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadius=2)
        .encode(
            x=alt.X("yearmonthdatehours(created_at):T", title=None),
            y=alt.Y("count():Q", title="Classifications"),
            color=alt.Color(
                "predicted_label:N", scale=LABEL_SCALE, legend=alt.Legend(orient="top", title=None)
            ),
            tooltip=[alt.Tooltip("yearmonthdatehours(created_at):T", title="When"), "count()"],
        )
        .properties(height=260)
    )


def decision_breakdown_chart(frame: pd.DataFrame) -> alt.Chart:
    """Summarise how many emails landed in each routing decision."""
    counts = frame["decision"].value_counts().reset_index()
    counts.columns = ["decision", "count"]
    return (
        alt.Chart(counts)
        .mark_bar(cornerRadius=4, height=34)
        .encode(
            x=alt.X("count:Q", title="Emails"),
            y=alt.Y("decision:N", sort="-x", title=None),
            color=alt.Color("decision:N", scale=DECISION_SCALE, legend=None),
            tooltip=["decision", alt.Tooltip("count:Q", format=",")],
        )
        .properties(height=180)
    )


def model_comparison_chart(comparison: list[dict]) -> alt.Chart:
    """Compare candidate models on their headline test metrics."""
    frame = pd.DataFrame(comparison).melt(
        id_vars="model",
        value_vars=["test_accuracy", "test_precision", "test_recall", "test_f1"],
        var_name="Metric",
        value_name="Score",
    )
    frame["Metric"] = (
        frame["Metric"].str.replace("test_", "", regex=False).str.replace("f1", "F1").str.title()
    )
    frame["model"] = frame["model"].str.replace("_", " ").str.title()

    return (
        alt.Chart(frame)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("Metric:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0.9, 1.0]), axis=alt.Axis(format="%")),
            color=alt.Color("model:N", legend=alt.Legend(orient="top", title=None)),
            xOffset="model:N",
            tooltip=["model", "Metric", alt.Tooltip("Score:Q", format=".4f")],
        )
        .properties(height=300)
    )
