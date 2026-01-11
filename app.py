"""
Database Migration Script
Normalizes all existing embeddings in the database
"""
import numpy as np
from pathlib import Path
import json

# Find the database files
POSSIBLE_PATHS = [
    Path("/app/storage"),
    Path("/app"),
    Path("./storage"),
    Path(".")
]

embeddings_file = None
labels_file = None

for path in POSSIBLE_PATHS:
    emb_path = path / "embeddings.npy"
    if emb_path.exists():
        embeddings_file = emb_path
        labels_file = path / "labels.npy"
        if not labels_file.exists():
            labels_file = path / "labels.json"
        print(f"Found database at: {path}")
        break

if not embeddings_file:
    print("ERROR: No embeddings.npy found!")
    exit(1)

# Load existing data
print(f"\nLoading from: {embeddings_file}")
embeddings = np.load(embeddings_file)
print(f"Original shape: {embeddings.shape}")
print(f"Original dtype: {embeddings.dtype}")

# Check if already normalized
norms = np.linalg.norm(embeddings, axis=1)
print(f"\nNorm statistics BEFORE:")
print(f"  Min: {norms.min():.4f}")
print(f"  Max: {norms.max():.4f}")
print(f"  Mean: {norms.mean():.4f}")
print(f"  Median: {np.median(norms):.4f}")

# Check if needs normalization
if norms.min() > 0.95 and norms.max() < 1.05:
    print("\n✅ Database is already normalized! No action needed.")
    exit(0)

print("\n⚠️  Database needs normalization!")

# Create backup
backup_file = embeddings_file.parent / f"{embeddings_file.stem}_backup.npy"
np.save(backup_file, embeddings)
print(f"\n✅ Backup saved to: {backup_file}")

# Normalize all embeddings
print("\nNormalizing embeddings...")
normalized_embeddings = []
for i, emb in enumerate(embeddings):
    norm = np.linalg.norm(emb)
    if norm > 0:
        normalized = emb / norm
    else:
        normalized = emb
        print(f"  WARNING: Embedding {i} has zero norm!")
    normalized_embeddings.append(normalized)

normalized_embeddings = np.array(normalized_embeddings)

# Verify normalization
new_norms = np.linalg.norm(normalized_embeddings, axis=1)
print(f"\nNorm statistics AFTER:")
print(f"  Min: {new_norms.min():.4f}")
print(f"  Max: {new_norms.max():.4f}")
print(f"  Mean: {new_norms.mean():.4f}")
print(f"  Median: {np.median(new_norms):.4f}")

# Test similarity before/after
print("\nTesting similarity changes...")
if len(embeddings) >= 2:
    # Old similarity (unnormalized)
    old_sim = np.dot(embeddings[0], embeddings[1])
    # New similarity (normalized)
    new_sim = np.dot(normalized_embeddings[0], normalized_embeddings[1])
    print(f"  Example similarity (0 vs 1):")
    print(f"    Before: {old_sim:.4f}")
    print(f"    After:  {new_sim:.4f}")

# Save normalized embeddings
print(f"\nSaving normalized embeddings to: {embeddings_file}")
np.save(embeddings_file, normalized_embeddings)

print("\n" + "="*60)
print("✅ MIGRATION COMPLETE!")
print("="*60)
print(f"\nBackup location: {backup_file}")
print(f"Original embeddings: {len(embeddings)}")
print(f"Normalized embeddings: {len(normalized_embeddings)}")
print("\nYou can now restart your API server.")
print("\nIf something goes wrong, restore from backup:")
print(f"  cp {backup_file} {embeddings_file}")
