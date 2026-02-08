
import tensorflow as tf
import numpy as np
import os

print("="*70)
print("TEST MODEL LOADING & INSPECTION")
print("="*70)

# Check files
print("\n1. CHECKING FILES...")
files_needed = ['model_embedding_aksen.keras', 'support_set.npy', 'support_labels.npy']
for f in files_needed:
    exists = os.path.exists(f)
    status = "✅" if exists else "❌"
    print(f"  {status} {f}")
    if exists and f.endswith('.npy'):
        data = np.load(f)
        print(f"      Shape: {data.shape}")

if not os.path.exists('model_embedding_aksen.keras'):
    print("\n❌ Model file tidak ditemukan!")
    exit(1)

# Define PrototypicalNetwork
print("\n2. DEFINING CUSTOM CLASS...")
@tf.keras.utils.register_keras_serializable(package="Custom")
class PrototypicalNetwork(tf.keras.Model):
    def __init__(self, embedding_model=None, **kwargs):
        super(PrototypicalNetwork, self).__init__(**kwargs)
        self.embedding_model = embedding_model
    
    def call(self, support_set, query_set, support_labels, n_way, training=None):
        return -tf.ones((1, n_way))  # Dummy
    
    def get_config(self):
        return super().get_config()

print("✅ PrototypicalNetwork defined")

# Load model
print("\n3. LOADING MODEL...")
try:
    custom_objects = {"PrototypicalNetwork": PrototypicalNetwork}
    model = tf.keras.models.load_model(
        "model_embedding_aksen.keras",
        custom_objects=custom_objects,
        compile=False
    )
    print("✅ Model loaded successfully")
    print(f"   Type: {type(model).__name__}")
    print(f"   Class: {model.__class__.__name__}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Inspect model
print("\n4. INSPECTING MODEL ATTRIBUTES...")
print("   Attributes (non-private):")
attrs = [a for a in dir(model) if not a.startswith('_')]
for attr in attrs[:20]:  # First 20
    try:
        val = getattr(model, attr)
        if not callable(val):
            print(f"     - {attr}: {type(val).__name__}")
    except:
        pass

# Check for embedding
print("\n5. CHECKING FOR EMBEDDING...")
embedding_found = False

if hasattr(model, 'embedding'):
    print(f"   ✅ model.embedding exists: {type(model.embedding)}")
    embedding_found = True
else:
    print("   ❌ model.embedding NOT found")

if hasattr(model, '_embedding_model'):
    print(f"   ✅ model._embedding_model exists: {type(model._embedding_model)}")
    embedding_found = True
else:
    print("   ❌ model._embedding_model NOT found")

if hasattr(model, 'embedding_model'):
    print(f"   ✅ model.embedding_model exists: {type(model.embedding_model)}")
    embedding_found = True
else:
    print("   ❌ model.embedding_model NOT found")

# Check layers
print("\n6. CHECKING LAYERS...")
if hasattr(model, 'layers'):
    print(f"   Total layers: {len(model.layers)}")
    for i, layer in enumerate(model.layers[:10]):  # First 10
        print(f"     [{i}] {layer.__class__.__name__}: {layer.name}")
        if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)):
            print(f"         ^^ This is a Model/Sequential!")
            embedding_found = True
else:
    print("   ❌ No layers attribute")

# Check weights
print("\n7. CHECKING WEIGHTS...")
if hasattr(model, 'weights'):
    print(f"   Total weights: {len(model.weights)}")
    for w in model.weights[:5]:
        print(f"     - {w.name}: {w.shape}")
else:
    print("   ❌ No weights")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
if embedding_found:
    print("✅ Embedding layer FOUND in model!")
else:
    print("❌ Embedding layer NOT FOUND - will need to rebuild")
    
    # Show all object attributes
    print("\nAll model.__dict__ keys:")
    for key in model.__dict__.keys():
        val = model.__dict__[key]
        print(f"  {key}: {type(val).__name__}")

print("\n" + "="*70)
print("TEST COMPLETED")
print("="*70)
