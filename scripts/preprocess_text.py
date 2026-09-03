#!/usr/bin/env python3
"""Generate ICLEA name/relation/description pickle features from raw TSV text.

Input TSVs contain ``integer_id<TAB>text``. Entity IDs must use the HY/ICLEA
ID space consumed by ``valid.ref`` and ``test.ref``; relation IDs must use the
JAPE relation space consumed by ``jape_triples_*``.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from urllib.parse import unquote

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity-names', type=Path, required=True)
    parser.add_argument('--relation-names', type=Path, required=True)
    parser.add_argument('--descriptions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--side', choices=('1', '2'), required=True)
    parser.add_argument('--setting', choices=('original', 'translated'), default='original')
    parser.add_argument('--model', default='sentence-transformers/LaBSE')
    parser.add_argument('--model-revision', default=None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--description-characters', type=int, default=512)
    parser.add_argument('--allow-missing-descriptions', action='store_true')
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path):
    values = {}
    with path.open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            fields = line.rstrip('\n').split('\t', 1)
            if len(fields) != 2:
                raise ValueError('{}:{} is not ID<TAB>text'.format(path, line_number))
            identifier = int(fields[0])
            if identifier in values:
                raise ValueError('{} repeats ID {}'.format(path, identifier))
            values[identifier] = fields[1]
    return values


def clean_name(value):
    value = unquote(value.strip())
    value = value.rsplit('/', 1)[-1].rsplit('#', 1)[-1]
    return ' '.join(value.replace('_', ' ').split())


def encode(model, values, batch_size, normalize=True):
    import numpy as np

    embeddings = model.encode(
        values,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[1] != 768:
        raise ValueError('Expected 768-dimensional embeddings, got {}'.format(embeddings.shape))
    return embeddings


def encode_mean_tokens(model, values, batch_size):
    """Implement the paper's name/relation mean-token pooling equation."""
    import numpy as np
    import torch

    chunks = []
    transformer = model[0].auto_model
    transformer.eval()
    for start in range(0, len(values), batch_size):
        tokenized = model.tokenizer(
            values[start:start + batch_size],
            padding=True,
            truncation=True,
            max_length=model.get_max_seq_length(),
            return_tensors='pt',
        )
        tokenized = {key: value.to(model.device) for key, value in tokenized.items()}
        with torch.no_grad():
            hidden = transformer(**tokenized, return_dict=True).last_hidden_state
            mask = tokenized['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        chunks.append(pooled.cpu().numpy().astype(np.float32))
    embeddings = np.concatenate(chunks, axis=0)
    if embeddings.ndim != 2 or embeddings.shape[1] != 768:
        raise ValueError('Expected 768-dimensional embeddings, got {}'.format(embeddings.shape))
    return embeddings


def archive(ids, embeddings):
    # Preserve the schema used by the published loader: ID -> [768-vector].
    return {identifier: [vector] for identifier, vector in zip(ids, embeddings)}


def write_pickle(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as f:
        pickle.dump(value, f, protocol=4)
    temporary.replace(path)


def main():
    args = parse_args()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit('Install requirements-preprocess.txt before running this script') from exc

    entity_names = read_tsv(args.entity_names)
    relation_names = read_tsv(args.relation_names)
    descriptions = read_tsv(args.descriptions)
    missing = sorted(set(entity_names) - set(descriptions))
    extra = sorted(set(descriptions) - set(entity_names))
    if extra:
        raise ValueError('Descriptions contain {} unknown entity IDs'.format(len(extra)))
    if missing and not args.allow_missing_descriptions:
        raise ValueError('Descriptions are missing {} entity IDs'.format(len(missing)))
    for identifier in missing:
        descriptions[identifier] = ''

    model_kwargs = {'device': args.device}
    if args.model_revision:
        model_kwargs['revision'] = args.model_revision
    model = SentenceTransformer(args.model, **model_kwargs)
    entity_ids = list(entity_names)
    relation_ids = list(relation_names)
    entity_vectors = encode_mean_tokens(
        model, [clean_name(entity_names[key]) for key in entity_ids], args.batch_size
    )
    relation_vectors = encode_mean_tokens(
        model, [clean_name(relation_names[key]) for key in relation_ids], args.batch_size
    )
    description_vectors = encode(
        model,
        [descriptions[key][: args.description_characters] for key in entity_ids],
        args.batch_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.setting == 'original':
        entity_file = args.output_dir / ('raw_LaBSE_emb_' + args.side + '.pkl')
        description_file = args.output_dir / ('desc_LaBSE_emb_' + args.side + '.pkl')
    else:
        entity_file = args.output_dir / ('transLaBSE_name_emb_' + args.side + '.pkl')
        description_file = args.output_dir / ('trans_desc_LaBSE_emb_' + args.side + '.pkl')
    relation_file = args.output_dir / ('jape_relation_emb_' + args.side + '.pkl')
    write_pickle(entity_file, archive(entity_ids, entity_vectors))
    write_pickle(description_file, archive(entity_ids, description_vectors))
    write_pickle(relation_file, archive(relation_ids, relation_vectors))

    manifest = {
        'schema_version': 1,
        'side': args.side,
        'setting': args.setting,
        'model': args.model,
        'model_revision': args.model_revision,
        'description_characters': args.description_characters,
        'name_relation_encoder': 'last_hidden_state attention-mask mean pooling then L2',
        'description_encoder': 'SentenceTransformer.encode(normalize_embeddings=True)',
        'name_cleanup': 'URL final component; percent decode; underscores to spaces',
        'counts': {
            'entities': len(entity_ids),
            'relations': len(relation_ids),
            'missing_descriptions_filled_empty': len(missing),
        },
        'inputs': {
            str(path): sha256(path)
            for path in (args.entity_names, args.relation_names, args.descriptions)
        },
        'outputs': {
            str(path): sha256(path)
            for path in (entity_file, relation_file, description_file)
        },
    }
    manifest_path = args.output_dir / 'preprocess_manifest_{}_{}.json'.format(
        args.setting, args.side
    )
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
