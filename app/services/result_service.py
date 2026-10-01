import json
from pathlib import Path


class ResultService:
    """
    Handles persistent storage and retrieval of ANPR results.
    """

    def __init__(self):
        project_root = Path(__file__).resolve().parents[2]

        self.results_dir = project_root / "data" / "results"

        self.results_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_result(self, result: dict) -> str:
        """
        Save an ANPR result as JSON.

        Returns:
            Result ID.
        """

        result_id = result["result_id"]

        result_path = (
            self.results_dir /
            f"{result_id}.json"
        )

        with open(
            result_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=4
            )

        return result_id

    def get_result(self, result_id: str):
        """
        Retrieve one result.
        """

        result_path = (
            self.results_dir /
            f"{result_id}.json"
        )

        if not result_path.exists():
            return None

        with open(
            result_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    def get_all_results(self):
        """
        Retrieve all stored results.
        """

        results = []

        for result_path in sorted(
            self.results_dir.glob("*.json"),
            reverse=True
        ):

            try:

                with open(
                    result_path,
                    "r",
                    encoding="utf-8"
                ) as file:

                    results.append(
                        json.load(file)
                    )

            except Exception:
                continue

        return results