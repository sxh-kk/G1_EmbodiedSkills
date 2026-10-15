#!/usr/bin/env python3
"""Read completed M2 development evidence; never auto-authorize engineering.

No simulator/model/GPU calls. Only explicit pipeline paths are opened. Runtime
source immutability is checked against the completed pipeline and audit; later
intentional source changes are reported separately and never rewrite old runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


class Evidence:
    def __init__(self):
        self.hashes = {}

    def pin(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_file(), f'Missing evidence: {path}')
        actual = digest(path)
        require(expected is None or actual == expected, f'Evidence SHA mismatch: {path}')
        require(str(path) not in self.hashes or self.hashes[str(path)] == actual,
                f'Evidence changed during analysis: {path}')
        self.hashes[str(path)] = actual
        return path

    def read(self, path, expected=None):
        path = self.pin(path, expected)
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == self.hashes[str(path)], f'Read race: {path}')
        return json.loads(data, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))

    def finish(self):
        require(all(digest(path) == value for path, value in self.hashes.items()),
                'Evidence/source changed while summarizing')
        return {'all_unchanged': True, 'files': dict(self.hashes), 'count': len(self.hashes)}


def source_paths(pipeline, manifest, audit):
    values = set(manifest['provenance']['sources'])
    for name, row in pipeline['inputs_and_sources']['files'].items():
        if name.startswith(('project/', 'sonic/', 'embodiedskills/')):
            values.add(row['path'])
    auditor_sources = audit.get('auditor_source_sha256', {})
    if isinstance(auditor_sources, dict):
        values.update(auditor_sources)
    elif isinstance(auditor_sources, str):
        # Current audits store one digest, not a path-to-digest mapping.
        # Recover the historical auditor path from that audit's pinned files.
        values.update(path for path, value in audit['evidence_hashes_before_after']['before'].items()
                      if value == auditor_sources and Path(path).name == 'm2_audit.py')
    return {str(Path(path).resolve()) for path in values}


def frozen_fingerprint(pipeline):
    values = {name: row['sha256'] for name, row in pipeline['inputs_and_sources']['files'].items()}
    combined = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
    require(combined == pipeline['inputs_and_sources']['combined_sha256'], 'Corrupt frozen input inventory')
    sources = {name: value for name, value in values.items()
               if name.startswith(('project/', 'sonic/', 'embodiedskills/'))}
    physical_core_names = ('isaac_python_wrapper', 'robot_asset', 'sonic_checkpoint',
        'sonic_checkpoint_configuration', 'sonic_fk_mjcf', 'sonic_sole_urdf')
    require(all(name in values for name in physical_core_names), 'Incomplete frozen physical controller/robot inputs')
    physical_core = {name: values[name] for name in physical_core_names}
    # Shard-specific paths/configurations vary across otherwise identical runs.
    # Changing executable sources or the protocol starts a distinct cohort.
    cohort = hashlib.sha256(json.dumps({'sources': sources, 'physical_core': physical_core,
        'protocol': pipeline['protocol']}, sort_keys=True).encode()).hexdigest()
    return combined, cohort


def verified_pipeline(path, evidence):
    pipeline = evidence.read(path)
    require(pipeline.get('schema') == 'g1_embodiedskills.m2_pipeline.v1', 'Unknown pipeline schema')
    require(pipeline.get('phase') == 'development', 'Only development pipelines are accepted')
    require(pipeline.get('completed') is True and pipeline.get('status') == 'complete',
            'Refusing to read any shard from an incomplete/running pipeline')
    require(pipeline.get('source_changes') == [], 'Missing/nonempty completed source-change check')
    require(pipeline.get('runs'), 'Completed pipeline has no physical shards')
    fingerprint, cohort = frozen_fingerprint(pipeline)
    groups, shards, later_source_drift = [], [], {}
    cost = dict(attempted_anchors=0, analysis_branches=0, physical_total_slot_rollouts=0,
                physical_padding_slot_rollouts=0, physical_extra_active_companion_rollouts=0,
                physical_active_branch_rows=0)
    for row in pipeline['runs']:
        require(row.get('status') == 'complete' and row.get('physical_audit_passed') is True
                and row.get('returncode') == 0, 'Shard physical process/audit did not complete')
        manifest = evidence.read(row['manifest'], row['manifest_sha256'])
        audit = evidence.read(row['audit'], row['audit_sha256'])
        require(manifest.get('completed') is True and audit.get('completed') is True
                and audit.get('passed') is True and audit.get('official_cpu_fk_verified') is True,
                'Missing completed official-FK physical audit')
        require(audit.get('manifest_sha256') == row['manifest_sha256'], 'Audit covers a different manifest')
        matching = [collection for collection in audit['collections']
                    if Path(collection['manifest_path']).resolve() == Path(row['manifest']).resolve()]
        require(len(matching) == 1 and matching[0]['passed'] is True
                and matching[0]['manifest_sha256'] == row['manifest_sha256'], 'Missing exact audited collection')
        integrity = audit['evidence_hashes_before_after']
        require(integrity.get('unchanged') is True and integrity['before'] == integrity['after'],
                'Physical audit evidence changed during its execution')
        recorded_sources = source_paths(pipeline, manifest, audit)
        for source, expected in manifest['provenance']['sources'].items():
            require(integrity['before'].get(str(Path(source).resolve())) == expected,
                    'Collection source was not pinned by its independent audit')
        # Recheck all historical artifact bytes. Intentionally newer executable
        # sources are not substituted into an old physical audit or hidden.
        for filename, expected in integrity['before'].items():
            filename = str(Path(filename).resolve())
            if filename in recorded_sources:
                if not Path(filename).is_file() or digest(filename) != expected:
                    later_source_drift[filename] = {'recorded_sha256': expected,
                        'current_sha256': digest(filename) if Path(filename).is_file() else None}
            else:
                evidence.pin(filename, expected)
        base = Path(row['manifest']).resolve().parent
        capsules = {capsule['id']: capsule for capsule in manifest['capsules']}
        for group in manifest['groups']:
            repeated = capsules[group['capsule_id']]['repeat_full_batch']
            expected = {(r, c) for r in range(3 if repeated else 1) for c in range(9)}
            actual = [(run['repeat'], run['candidate']) for run in group['runs']]
            require(len(actual) == len(set(actual)) and set(actual) == expected,
                    'Incomplete run inventory cannot be interpreted as physical censoring')
            groups.append({**group, '_base': base, '_cohort': cohort,
                '_pipeline': str(Path(path).resolve()), '_audited_evidence': integrity['before']})
        for key in cost:
            cost[key] += int(manifest[key])
        require(row['attempted_anchors'] == manifest['attempted_anchors']
                and row['analysis_branches'] == manifest['analysis_branches']
                and row['physical_total_slot_rollouts'] == manifest['physical_total_slot_rollouts'],
                'Pipeline counts differ from the audited manifest')
        shards.append({'name': row['name'], 'manifest': str(base/'manifest.json'),
            'manifest_sha256': row['manifest_sha256'], 'physical_audit': row['audit'],
            'physical_audit_sha256': row['audit_sha256'], 'physical_audit_passed': True,
            'official_cpu_fk_verified': True})
    return {'pipeline': str(Path(path).resolve()), 'pipeline_sha256': evidence.hashes[str(Path(path).resolve())],
        'started_at': pipeline['started_at'], 'finished_at': pipeline['finished_at'],
        'wall_seconds': pipeline['wall_seconds'], 'source_changes_during_run': [],
        'source_input_fingerprint': fingerprint, 'cohort': cohort, 'protocol': pipeline['protocol'],
        'later_executable_source_drift': later_source_drift, 'shards': shards,
        'physical_cost': cost}, groups


def body_contrasts(group, analysis):
    _, samples = analysis._run_arrays(group)
    result = {}
    for repeat in range(3):
        waist, valid = analysis._grid(samples, repeat, 'waist_change', 1)
        for body in (1, 2):
            value = float(waist[analysis.CELL_INDEX[0, body]]-waist[0]) if valid[body] and valid[0] else None
            result[repeat, body] = value
    return result


def summarize(paths, *, project, reference_public=None):
    sys.path.insert(0, str(Path(project).resolve()/'src'))
    from g1_embodiedskills import m2_analysis as analysis
    evidence = Evidence()
    evidence.pin(__file__)
    evidence.pin(analysis.__file__)
    pipelines, groups = [], []
    unique = [Path(path).resolve() for path in paths]
    require(len(unique) == len(set(unique)), 'Duplicate input pipeline')
    for path in unique:
        pipeline, local = verified_pipeline(path, evidence)
        pipelines.append(pipeline)
        groups.extend(local)
    identities = [(g['_cohort'], g['condition'], g['seed'], g['anchor_step']) for g in groups]
    require(len(identities) == len(set(identities)), 'Duplicate group within one source/protocol cohort')
    repeated = {}
    contrasts = {}
    for group in groups:
        key = (group['_cohort'], group['condition'], group['seed'], group['anchor_step'])
        contrasts[key] = body_contrasts(group, analysis)
        if group.get('noise_anchor'):
            noise_key = group['_cohort'], group['family'], group['anchor_step']
            require(noise_key not in repeated, 'Multiple noise references in the same cohort/family/phase')
            repeated[noise_key] = group
    output, blockers = [], []
    for group in groups:
        metadata = group.get('candidate_metadata', {})
        own = contrasts[group['_cohort'], group['condition'], group['seed'], group['anchor_step']]
        reference = repeated.get((group['_cohort'], group['family'], group['anchor_step']))
        rows = []
        for body, sign in ((1, 1), (2, -1)):
            values = [own[r, body] for r in range(3)]
            noise_values = ([contrasts[reference['_cohort'], reference['condition'], reference['seed'], reference['anchor_step']][r, body]
                             for r in range(3)] if reference else [None]*3)
            noise = analysis.centered_repeat_noise([np.nan if x is None else x for x in noise_values])
            delta = values[0]
            threshold = max(3*noise, 1e-4) if noise is not None else None
            correct = bool(delta is not None and threshold is not None and sign*delta > threshold)
            rows.append({'body_index': body, 'body_sign': sign,
                'contrasts_rad': values, 'first_pass_signed_response_rad': sign*delta if delta is not None else None,
                'centered_repeat_noise_rad': noise, 'direction_threshold_rad': threshold,
                'first_pass_direction_correct_above_floor_and_noise': correct,
                'noise_reference_seed': reference['seed'] if reference else None,
                'noise_reference_condition': reference['condition'] if reference else None,
                'noise_reference_amplitudes_match': bool(reference is not None
                    and metadata.get('effective_amplitude_m') == reference.get('candidate_metadata', {}).get('effective_amplitude_m')
                    and metadata.get('effective_body_pitch_rad') == reference.get('candidate_metadata', {}).get('effective_body_pitch_rad'))})
        feasible = group.get('candidate_feasible')
        grid = bool(group.get('accepted') and isinstance(feasible, list) and len(feasible) == 9 and all(feasible))
        row = {key: group[key] for key in ('id', 'condition', 'family', 'seed', 'anchor_step', 'slot', 'capsule_id', 'accepted', 'rejections')}
        row.update(cohort=group['_cohort'], pipeline=group['_pipeline'], candidate_feasible=feasible,
            complete_feasible_grid=grid,
            **{key: metadata.get(key) for key in ('requested_amplitude_m', 'requested_body_pitch_rad',
                'effective_amplitude_m', 'effective_body_pitch_rad', 'common_scale')},
            analysis_runs=sum(run.get('analysis_included', True) for run in group['runs']),
            valid_analysis_runs=sum(run.get('analysis_included', True) and run.get('valid') for run in group['runs']),
            signed_body_response_half_second=rows,
            both_body_directions_observed_above_noise=all(item['first_pass_direction_correct_above_floor_and_noise'] for item in rows))
        output.append(row)
        if not grid:
            blockers.append({'group': group['id'], 'reason': 'Nine-cell feasible/accepted grid unavailable'})
        if not row['both_body_directions_observed_above_noise']:
            blockers.append({'group': group['id'], 'reason': 'Both signed body responses above matched R3 noise/floor not demonstrated'})
    def coverage_for(selected):
        return {condition: {str(anchor): {'groups': sum(g['condition'] == condition and g['anchor_step'] == anchor for g in selected),
            'accepted_grids': sum(g['condition'] == condition and g['anchor_step'] == anchor and g['complete_feasible_grid'] for g in selected),
            'both_body_directions': sum(g['condition'] == condition and g['anchor_step'] == anchor and g['both_body_directions_observed_above_noise'] for g in selected)}
            for anchor in analysis.ANCHOR_STEPS} for condition in analysis.CONDITIONS}
    coverage = coverage_for(output)
    coverage_by_cohort = {cohort: coverage_for([group for group in output if group['cohort'] == cohort])
                          for cohort in sorted({group['cohort'] for group in output})}
    for cohort, matrix in coverage_by_cohort.items():
        missing = [(condition, anchor) for condition in analysis.CONDITIONS for anchor in analysis.ANCHOR_STEPS
                   if matrix[condition][str(anchor)]['groups'] == 0]
        if missing:
            blockers.append({'cohort': cohort, 'reason': 'Development condition/anchor coverage incomplete', 'missing': missing})
    report = {'schema': 'g1_embodiedskills.m2_engineering_summary.v1', 'completed': True,
        'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'scope': 'Read-only engineering diagnostics; physical record audit is separate from engineering acceptance',
        'physical_record_audit_passed': True, 'engineering_passed': False if blockers else None,
        'engineering_acceptance_policy': 'Never auto-set true; group blockers are conservative diagnostics; '
            'formal release follows the fixed rule in docs/M2-工程验收与正式采集协议.md',
        'body_definition': 'D_b=[waist_b(t)-waist_b(anchor)]-[waist_0(t)-waist_0(anchor)], u=0, t=+.5s',
        'direction_gate': 'sign(b)*D_b > max(3*centered_R3_paired_noise, 1e-4 rad)',
        'noise_definition': 'sqrt(sum_r ||D_r-mean(D)||^2/(R-1)), R=3; same cohort/family/anchor only; missing is not zero',
        'pipelines': pipelines, 'groups': output, 'coverage': coverage,
        'coverage_by_cohort': coverage_by_cohort,
        'pooling_scope': 'Noise and coverage gates never pool different executable/physical-input/protocol cohorts; top-level coverage is an inventory only',
        'engineering_blockers': blockers,
        'cost_totals': {key: sum(row['physical_cost'][key] for row in pipelines) for key in pipelines[0]['physical_cost']}}
    if reference_public:
        reference = evidence.read(reference_public)
        required = ('attempted_anchors', 'analysis_branches', 'physical_total_slot_rollouts',
                    'physical_padding_slot_rollouts', 'physical_extra_active_companion_rollouts')
        for key in required:
            require(report['cost_totals'][key] == reference[key], f'Public metric mismatch: {key}')
        require(report['engineering_passed'] == reference['engineering_passed'], 'Public engineering status mismatch')
        for expected in reference['groups']:
            matches = [g for g in output if g['seed'] == expected['seed']]
            require(len(matches) == 1, 'Public comparison seed is not unique')
            actual = matches[0]
            for key in ('accepted', 'rejections', 'candidate_feasible', 'effective_amplitude_m',
                        'effective_body_pitch_rad', 'common_scale', 'analysis_runs', 'valid_analysis_runs'):
                require(actual[key] == expected[key], f'Public group metric mismatch: {key}')
            for expected_body in expected.get('signed_body_response_half_second', []):
                actual_body = actual['signed_body_response_half_second'][expected_body['body_index']-1]
                for key, value in expected_body.items():
                    require(actual_body[key] == value, f'Public body metric mismatch: {key}')
        report['reference_public_comparison'] = {'matched': True, 'path': str(Path(reference_public).resolve()),
            'sha256': evidence.hashes[str(Path(reference_public).resolve())]}
    report['evidence_integrity'] = evidence.finish()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipelines', nargs='+', type=Path, required=True)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Repository root (default: parent of analysis/)')
    parser.add_argument('--reference-public', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    require(not args.output.exists(), 'Refusing to overwrite an existing report')
    report = summarize(args.pipelines, project=args.project, reference_public=args.reference_public)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'completed': True, 'groups': len(report['groups']),
        'physical_record_audit_passed': True, 'engineering_passed': report['engineering_passed'],
        'blockers': len(report['engineering_blockers']),
        'reference_public_matched': report.get('reference_public_comparison', {}).get('matched'),
        'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
