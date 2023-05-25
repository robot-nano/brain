import csv
import logging

logger = logging.getLogger(__name__)


def load_data_csv(csv_path):
    with open(csv_path, "r") as csvfile:
        result = {}
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        for row in reader:
            try:
                data_id = row["ID"]
                del row["ID"]
            except KeyError:
                raise KeyError(
                    "CSV has to have an 'ID' field, with unique ids"
                    " for all data points"
                )
            if data_id in result:
                raise ValueError(f"Duplicate id: {data_id}")

            # Duration:
            if "duration" in row:
                row["duration"] = float(row["duration"])
            result[data_id] = row
    return result
