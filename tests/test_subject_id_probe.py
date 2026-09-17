from src.finetune.linear_probe_eval import load_pretrained_encoder
from src.visualization.embedding_diagnostics import gather_pretrain_embeddings
from src.analysis.subject_id_probe import run_subject_id_probe
from src.utils.device import get_device

device = get_device()
encoder = load_pretrained_encoder("checkpoints/encoder_multi_full_e25.pt", device)
embeddings, classes, subjects, datasets = gather_pretrain_embeddings(
    encoder, ["bnci2014_001", "physionet_mi"], device
)

result = run_subject_id_probe(embeddings, subjects)
print(f"Subject-ID probe: {result['n_subjects']} subjects, chance={result['chance_level']:.3f}")
print(f"Mean accuracy: {result['mean_accuracy']:.3f} (+/- {result['std_accuracy']:.3f})")
print(f"Fold scores: {[round(s,3) for s in result['fold_scores']]}")