import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from addict_probe.bridge import load_runtime_trace_jsonl, rows_to_feature_matrix

if __name__ == '__main__':
    rows = load_runtime_trace_jsonl(sys.argv[1])
    x = rows_to_feature_matrix(rows)
    print({'rows': len(rows), 'feature_shape': tuple(x.shape)})
