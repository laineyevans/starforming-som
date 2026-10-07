import pandas as pd


def load_imageproperties(imageproperties_file):
    """
    Load the clump-to-internal-ID mapping from imageproperties.dat.
    Only Name and IntID are required.
    """

    # Explicit column list based on byte-by-byte description
    colnames = [
        "Num", "Name", "OrigName", "IntID", "GLON", "GLAT", "subflag",
        "Vlsr", "Dist", "e_Dist", "Mass", "e_Mass", "Lum", "e_Lum",
        "Tdust", "Tbol", "Sdens", "Ecl"
    ]

    df = pd.read_csv(
        imageproperties_file,
        sep=r"\s+",
        header=None,
        names=colnames,
        comment="#"
    )

    # Ensure IntID is treated as string for filenames
    df["IntID"] = df["IntID"].astype(str)

    return df[["Name", "IntID"]]


def load_core_table(core_file):
    """
    Load table3.dat containing core coordinates and ellipse parameters.
    Expects columns: Name, Seq, RAdeg, DEdeg, Diam1, Diam2, PAs.
    """

    df = pd.read_csv(core_file, sep=r"\s+")
    return df


def merge_core_and_ID(core_df, id_df):
    """
    Merge table3 cores with imageproperties IDs using 'Name'.
    """
    merged = pd.merge(core_df, id_df, on="Name", how="inner")
    return merged


def write_region_files(merged_df, output_dir="."):
    """
    Write DS9 region files for each IntID.
    """

    for int_id in merged_df["IntID"].unique():

        outfile = f"{output_dir}/{int_id}_cores_ds9.reg"
        df_sub = merged_df[merged_df["IntID"] == int_id]

        with open(outfile, "w") as f:
            for _, row in df_sub.iterrows():

                ra = row["RAdeg"]
                dec = row["DEdeg"]
                r1 = float(row["Diam1"]) / 2.0
                r2 = float(row["Diam2"]) / 2.0
                pa = row["PAs"]
                seq = row["Seq"]

                line = (
                    f"j2000; ellipse {ra}d {dec}d {r1}\" {r2}\" {pa}d "
                    f"# text={{{seq}}} color=cyan\n"
                )
                f.write(line)


def run_core_identifier(core_file, imageproperties_file, output_dir="."):
    """
    Full pipeline entry point for use inside run_pipeline.py.
    """
    core_df = load_core_table(core_file)
    id_df = load_imageproperties(imageproperties_file)
    merged = merge_core_and_ID(core_df, id_df)
    write_region_files(merged, output_dir)
    return merged
