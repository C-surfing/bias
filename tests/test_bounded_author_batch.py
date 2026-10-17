"""Service lifecycle checks: shutdown targets only the process group we created."""
import signal
import subprocess
import unittest
from unittest.mock import Mock,call,patch

from scripts.run_bounded_author_batch import request_stop,stop_group


class BatchShutdownTests(unittest.TestCase):
    def setUp(self):
        # Windows does not expose Linux SIGKILL; this suite simulates the server.
        linux_signal=patch("scripts.run_bounded_author_batch.signal.SIGKILL",9,create=True)
        linux_signal.start()
        self.addCleanup(linux_signal.stop)

    def test_already_exited_group_never_targets_another_process(self):
        process=Mock(pid=12345)
        with patch("scripts.run_bounded_author_batch.os.killpg",create=True,side_effect=ProcessLookupError) as kill:
            stop_group(process)
        kill.assert_called_once_with(12345,signal.SIGTERM)
        process.wait.assert_not_called()

    def test_parent_exit_does_not_leave_its_workers_running(self):
        process=Mock(pid=12345)
        with patch("scripts.run_bounded_author_batch.os.killpg",create=True) as kill:
            stop_group(process)
        self.assertEqual(kill.call_args_list,[call(12345,signal.SIGTERM),call(12345,0),call(12345,signal.SIGKILL)])
        process.wait.assert_called_once_with(timeout=45)

    def test_unresponsive_group_is_killed_after_bounded_grace_period(self):
        process=Mock(pid=12345)
        process.wait.side_effect=[subprocess.TimeoutExpired("owned child",45),0]
        with patch("scripts.run_bounded_author_batch.os.killpg",create=True,side_effect=[None,None,ProcessLookupError]) as kill:
            stop_group(process)
        self.assertEqual(kill.call_args_list,[call(12345,signal.SIGTERM),call(12345,signal.SIGKILL),call(12345,0)])
        self.assertEqual(process.wait.call_args_list,[call(timeout=45),call(timeout=15)])

    def test_stop_signal_unwinds_into_finally_cleanup(self):
        cleaned=[]
        with self.assertRaises(SystemExit):
            try:request_stop(signal.SIGTERM,None)
            finally:cleaned.append(True)
        self.assertEqual(cleaned,[True])


if __name__=="__main__":unittest.main()
