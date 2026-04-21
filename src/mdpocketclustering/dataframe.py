import pandas as pd


def build_dataframe(all_sims):
    rows = []

    for sim in all_sims:
        rows.append({
            "mutation": sim.mutation,
            "replicate": sim.replicate,
            "ATP": bool(sim.has_atp),
            "MG": bool(sim.has_mg),
            "binding_energy": sim.binding_energy,
        })

    return pd.DataFrame(rows)


def clean_binding_dataframe(df):
    df = df.dropna(subset=["binding_energy"]).copy()

    df["condition"] = (
        df["ATP"].map({True: "ATP", False: "ADP"})
        + " | Mg="
        + df["MG"].map({True: "yes", False: "no"})
    )

    return df
