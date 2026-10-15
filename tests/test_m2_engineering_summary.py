"""Tool contracts only; generated numbers below are not robot experiments."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_PROJECT = Path(os.environ.get('G1_PROJECT_ROOT', PROJECT)).resolve()
sys.path.insert(0, str(PROJECT/'analysis'))
sys.path.insert(0, str(SOURCE_PROJECT/'src'))
import summarize_m2_engineering as tool
from g1_embodiedskills import m2_analysis as analysis


class SummaryContracts(unittest.TestCase):
    def test_running_input_is_rejected_before_any_shard_read(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'pipeline.json'
            path.write_text(json.dumps({'schema': 'g1_embodiedskills.m2_pipeline.v1',
                'phase': 'development', 'completed': False, 'status': 'running',
                'runs': [{'manifest': '/should-never-be-opened'}]}))
            evidence = tool.Evidence()
            with self.assertRaisesRegex(ValueError, 'incomplete/running'):
                tool.verified_pipeline(path, evidence)
            self.assertEqual(set(evidence.hashes), {str(path.resolve())})

    def test_missing_or_nonempty_source_change_check_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'pipeline.json'
            for changes in (None, ['changed-source']):
                path.write_text(json.dumps({'schema': 'g1_embodiedskills.m2_pipeline.v1',
                    'phase': 'development', 'completed': True, 'status': 'complete',
                    'source_changes': changes}))
                with self.assertRaisesRegex(ValueError, 'source-change'):
                    tool.verified_pipeline(path, tool.Evidence())

    def test_evidence_change_cannot_pass_on_a_previously_cached_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'evidence.json'
            path.write_text('{}')
            evidence = tool.Evidence()
            evidence.pin(path)
            path.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                evidence.finish()

    def test_actual_waist_contrast_subtracts_each_runs_own_anchor(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runs, audited = [], {}
            for repeat in range(3):
                for cell in range(9):
                    anchor = .005*(1+repeat*9+cell)
                    direction = (0, 1, -1)[cell % 3]
                    response = direction*(.03+.01*(repeat-1))
                    path = root/f'sample-{repeat}-{cell}.npz'
                    np.savez(path, waist=np.repeat(anchor+.2+response, 4),
                        valid=np.ones(4, bool), offset_steps=np.array([10,25,50,150]))
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    audited[str(path)] = digest
                    runs.append({'candidate': cell, 'repeat': repeat, 'analysis_included': True,
                        'valid': True, 'samples_archive': path.name, 'samples_sha256': digest,
                        'anchor_waist_rad': anchor})
            group = {'accepted': True, 'id': 'synthetic-contract', '_base': root,
                     '_audited_evidence': audited, 'runs': runs}
            contrast = tool.body_contrasts(group, analysis)
            np.testing.assert_allclose([contrast[r,1] for r in range(3)], [.02,.03,.04], atol=1e-16)
            np.testing.assert_allclose([contrast[r,2] for r in range(3)], [-.02,-.03,-.04], atol=1e-16)
            self.assertAlmostEqual(analysis.centered_repeat_noise([contrast[r,1] for r in range(3)]), .01)


if __name__ == '__main__':
    unittest.main()
