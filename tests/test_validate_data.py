import unittest

from src.validate_data import validate_rows


class ValidateRowsTests(unittest.TestCase):
    def setUp(self):
        self.sessions = ["2026-01-02", "2026-01-05"]  # Friday and Monday
        self.symbols = ["NVDA", "AMD"]
        self.complete = [
            {"date": date, "symbol": symbol, "adjusted_close": "100.25"}
            for date in self.sessions for symbol in self.symbols
        ]

    def test_complete_data_does_not_flag_weekend(self):
        self.assertEqual(validate_rows(self.complete, self.sessions, self.symbols), {
            "missing": [], "duplicates": [], "invalid_prices": [],
        })

    def test_missing_pair_is_reported(self):
        rows = self.complete[:-1]
        self.assertEqual(validate_rows(rows, self.sessions, self.symbols)["missing"],
                         ["2026-01-05 AMD"])

    def test_duplicate_pair_is_reported_without_dropping_it(self):
        rows = self.complete + [dict(self.complete[0])]
        self.assertEqual(validate_rows(rows, self.sessions, self.symbols)["duplicates"],
                         ["2026-01-02 NVDA"])

    def test_nonpositive_nonnumeric_and_nonfinite_prices(self):
        rows = [dict(row) for row in self.complete]
        for row, value in zip(rows, ["0", "-2", "not-a-number", "NaN"]):
            row["adjusted_close"] = value
        self.assertEqual(validate_rows(rows, self.sessions, self.symbols)["invalid_prices"],
                         sorted(f"{row['date']} {row['symbol']}" for row in rows))

    def test_does_not_mutate_input(self):
        rows = [dict(row) for row in self.complete]
        before = [dict(row) for row in rows]
        validate_rows(rows, self.sessions, self.symbols)
        self.assertEqual(rows, before)


if __name__ == "__main__":
    unittest.main()
