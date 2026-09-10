import unittest
from unittest.mock import patch

import run


class RunScriptTests(unittest.TestCase):
    def test_build_streamlit_argv(self):
        self.assertEqual(
            run.build_streamlit_argv(["--server.headless", "true"]),
            ["streamlit", "run", "app.py", "--server.headless", "true"],
        )

    def test_main_delegates_to_streamlit_cli(self):
        with patch("run.stcli.main", return_value=0) as mocked_main:
            exit_code = run.main(["--server.port", "8502"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            run.sys.argv,
            ["streamlit", "run", "app.py", "--server.port", "8502"],
        )
        mocked_main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
