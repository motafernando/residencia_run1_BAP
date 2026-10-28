"""Download pinned upstream files and prepare auditable WinoBias smoke inputs.

Only the Python standard library is required. This command does not load an LLM.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request

ROOT = Path(__file__).resolve().parent
REVISION = '0bce984dd081bbc10b0622f326727a024c607895'
UPSTREAM = 'https://raw.githubusercontent.com/uclanlp/corefBias/' + REVISION + '/'
FILES = ['LICENSE', 'README.md', 'WinoBias/wino/readme.md',
         'WinoBias/wino/anonymize.py', 'WinoBias/wino/word_swapper.py']
FILES += ['WinoBias/wino/data/' + name for name in
          ['Readme.md', 'female_occupations.txt', 'male_occupations.txt']]
FILES += [f'WinoBias/wino/data/{stereo}_stereotyped_type{kind}.txt.{split}'
          for split in ['dev', 'test'] for kind in [1, 2] for stereo in ['pro', 'anti']]
PRONOUNS = {'he', 'she', 'him', 'her', 'his', 'himself', 'herself', 'hers'}
FEMALE_PRONOUNS = {'she', 'her', 'hers', 'herself'}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def parse_line(line, filename, occupations):
    number, marked = line.split(' ', 1)
    spans = [s.strip() for s in re.findall(r'\[([^\[\]]+)\]', marked)]
    pronouns = [s.lower() for s in spans if s.lower() in PRONOUNS]
    antecedents = [s for s in spans if s.lower() not in PRONOUNS]
    if len(antecedents) != 1 or not pronouns:
        raise ValueError(f'Unexpected annotation: {filename}:{number}: {marked}')
    pronoun = pronouns[0]
    if len({'female' if p in FEMALE_PRONOUNS else 'male' for p in pronouns}) != 1:
        raise ValueError(f'Mixed annotated genders: {filename}:{number}')
    gold_span = antecedents[0]
    gold = re.sub(r'^(?:the|a|an)\s+', '', gold_span, flags=re.I).lower()
    sentence = marked.replace('[', '').replace(']', '')
    # Longest occupations first and whole-word matching preserve multiword names.
    matches = []
    for occupation in sorted(occupations, key=len, reverse=True):
        for match in re.finditer(r'(?<!\w)' + re.escape(occupation) + r'(?!\w)', sentence, re.I):
            if not any(match.start() < end and start < match.end() for start, end, _ in matches):
                matches.append((match.start(), match.end(), occupation))
    matches.sort()
    all_occupations = list(dict.fromkeys(m[2] for m in matches))
    candidates = all_occupations[:2]
    if len(candidates) != 2 or gold not in candidates:
        raise ValueError(f'Unexpected candidates: {filename}:{number}: {candidates}, gold={gold}')
    return {'id': f'{filename}:{number}', 'source_file': filename, 'source_line_id': int(number),
            'split': filename.rsplit('.', 1)[-1], 'type': int(re.search(r'type(\d)', filename)[1]),
            'stereotype': 'anti' if filename.startswith('anti') else 'pro',
            'gender': 'female' if pronoun in FEMALE_PRONOUNS else 'male',
            'pronoun': pronoun, 'annotated_pronouns': pronouns,
            'pronoun_policy': 'first annotated pronoun; all annotated pronouns retained',
            'sentence': sentence, 'marked_sentence': marked,
            'candidates': candidates, 'occupations_mentioned': all_occupations,
            'candidate_text': {c: next(sentence[start:end] for start, end, label in matches if label == c)
                               for c in candidates},
            'candidate_policy': 'first two distinct occupation mentions; additional mentions retained for review',
            'needs_review': len(all_occupations) != 2 or len(pronouns) != 1,
            'gold': gold}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true', help='Use already downloaded raw files')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    raw_root = ROOT / 'data' / 'raw' / 'corefBias' / REVISION
    previous = ROOT / 'data' / 'prepared' / 'inventory.json'
    known = {}
    if previous.exists():
        known = {e['path']: e['sha256'] for e in json.loads(previous.read_text(encoding='utf-8'))['files']}
    entries = []
    for filename in FILES:
        target = raw_root / filename
        if not target.exists():
            if args.offline:
                raise FileNotFoundError(target)
            request = urllib.request.Request(UPSTREAM + filename, headers={'User-Agent': 'bias-reproduction-research'})
            raw = urllib.request.urlopen(request, timeout=45).read()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        raw = target.read_bytes()
        if filename in known and digest(raw) != known[filename]:
            raise ValueError(f'Raw file differs from recorded inventory: {filename}')
        entries.append({'path': filename, 'url': UPSTREAM + filename,
                        'bytes': len(raw), 'sha256': digest(raw)})
    data_root = raw_root / 'WinoBias' / 'wino' / 'data'
    occupations = []
    for gender in ['female', 'male']:
        occupations += [s.strip().lower() for s in (data_root / f'{gender}_occupations.txt').read_text(encoding='utf-8').splitlines() if s.strip()]
    records = []
    for path in sorted(data_root.glob('*stereotyped_type*.txt.*')):
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                records.append(parse_line(line, path.name, occupations))
    counts = dict(collections.Counter(r['source_file'] for r in records))
    groups = collections.defaultdict(list)
    for record in records:
        if record['split'] == 'test' and not record['needs_review']:
            groups[(record['type'], record['stereotype'], record['gender'])].append(record)
    # Three examples in each of eight strata: 2 types x 2 stereotypes x 2 genders.
    rng = random.Random(args.seed)
    sample = []
    for group in sorted(groups):
        sample += rng.sample(groups[group], 3)
    rng.shuffle(sample)
    normalized = ROOT / 'data' / 'prepared'
    write_json(normalized / 'inventory.json', {
        'repository': 'https://github.com/uclanlp/corefBias', 'revision': REVISION,
        'revision_selection': 'Last upstream commit before arXiv v1; not an author-confirmed dataset revision',
        'license_file': str((raw_root / 'LICENSE').relative_to(ROOT).as_posix()),
        'files': entries, 'counts': counts, 'total': len(records),
        'review_rows': [r['id'] for r in records if r['needs_review']],
        'occupations': occupations, 'occupation_count': len(set(occupations)),
        'paper_count': 3160, 'count_difference': len(records) - 3160,
        'count_note': 'All upstream rows preserved; no trimming to match the published total.'})
    write_json(normalized / 'all_records.json', records)
    write_json(normalized / 'smoke24.json', {
        'seed': args.seed, 'convention': 'Local smoke sample: three rows per type/stereotype/gender stratum, test only',
        'dataset_revision': REVISION, 'records': sample})
    print(json.dumps({'status': 'data_prepared_no_inference', 'total': len(records),
                      'counts': counts, 'occupations': len(set(occupations)),
                      'sample': len(sample)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
