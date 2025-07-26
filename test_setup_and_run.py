import pytest
import subprocess
import sys
from unittest.mock import patch, MagicMock
from install_requirements import install_requirements, run_initial_steps

def test_install_requirements():
    with patch('subprocess.check_call') as mock_check_call:
        install_requirements()
        mock_check_call.assert_called_once_with([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])

def test_run_initial_steps(capsys):
    run_initial_steps()
    captured = capsys.readouterr()
    assert "Running initial steps..." in captured.out

@patch('subprocess.check_call')
@patch('builtins.print')
def test_main(mock_print, mock_check_call):
    from install_requirements import main
    main()
    assert mock_check_call.called
    mock_print.assert_any_call("Installing required packages...")
    mock_print.assert_any_call("Running initial setup steps...")
    mock_print.assert_any_call("Setup complete. Running main script...")
