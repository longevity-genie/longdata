from polars import DataFrame
import polars as pl

def write_data(frame: DataFrame, max_rows: int = 8) -> str:
    res = ""
    for col in frame.get_columns():
        res += col.name + " ; "
    res = res[:-3] + "\n"
    frame = frame.with_columns(pl.all().fill_null("unknown"))
    rows = frame.rows()
    for row in rows[:min(len(rows), max_rows)]:
        for cel in row:
            res += str(cel) + " ; "
        res = res[:-3] + "\n"

    return res