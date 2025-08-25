# ----------------------------
# Diagnostic: Check model files
# ----------------------------
import os 
print("🔍 Looking for model files...")
model_dir = "model"
if os.path.exists(model_dir):
    print(f"📁 Contents of '{model_dir}' folder:")
    for f in os.listdir(model_dir):
        print(f"   - {f}")
else:
    print(f"❌ Folder '{model_dir}' does not exist")

print("📁 Contents of current directory:")
for f in os.listdir():
    if 'metal' in f.lower() or 'model' in f.lower() or '.h5' in f or '.keras' in f:
        print(f"   - {f}")