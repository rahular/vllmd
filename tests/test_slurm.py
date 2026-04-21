"""Tests for slurm utilities."""

import subprocess
import pytest
from unittest.mock import MagicMock, patch

from vllmd.slurm import sbatch, squeue, scancel, slurm_available, SlurmError


class TestSbatch:
    def test_returns_job_id_on_success(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Submitted batch job 1234567\n",
                stderr="",
            )
            job_id = sbatch("#!/bin/bash\necho hello")
        assert job_id == "1234567"

    def test_raises_on_nonzero_exit(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="sbatch: error: AssocMaxSubmitJobLimit",
            )
            with pytest.raises(SlurmError, match="sbatch failed"):
                sbatch("#!/bin/bash\necho hello")

    def test_raises_when_job_id_not_in_output(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="some unexpected output\n",
                stderr="",
            )
            with pytest.raises(SlurmError, match="Could not parse job ID"):
                sbatch("#!/bin/bash\necho hello")


class TestSqueue:
    def test_parses_output(self):
        sample = (
            "1234567|RUNNING|None|nid001\n"
            "1234568|PENDING|Priority|N/A\n"
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=sample, stderr=""
            )
            rows = squeue(["1234567", "1234568"])

        assert len(rows) == 2
        assert rows[0]["job_id"] == "1234567"
        assert rows[0]["state"] == "RUNNING"
        assert rows[0]["node"] == "nid001"
        assert rows[1]["state"] == "PENDING"

    def test_empty_output(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            rows = squeue(["9999999"])
        assert rows == []


class TestScancel:
    def test_calls_scancel_with_job_ids(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            scancel(["1234567", "1234568"])
        call_args = mock_run.call_args[0][0]
        assert "scancel" in call_args
        assert "1234567" in call_args
        assert "1234568" in call_args

    def test_noop_on_empty_list(self):
        with patch("subprocess.run") as mock_run:
            scancel([])
        mock_run.assert_not_called()

    def test_tolerates_nonzero_exit(self):
        """scancel exits non-zero for already-finished jobs — should not raise."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="scancel: error: Invalid job id"
            )
            scancel(["9999999"])  # no exception


class TestSlurmAvailable:
    def test_true_when_sbatch_in_path(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="/usr/bin/sbatch\n")
            assert slurm_available() is True

    def test_false_when_not_found(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            assert slurm_available() is False
